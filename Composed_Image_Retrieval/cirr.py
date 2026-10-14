"""REST evaluation for CIRR composed image retrieval."""
import argparse
import os
from tqdm import tqdm
import numpy as np
import random
import ruamel.yaml as yaml
import time
import datetime
import json
import tempfile
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import utils
from lavis.models import load_model_and_preprocess
from models.set_model import *
from data_utils import CIRRDataset, targetpad_transform, base_path
from utils import device, extract_index_blip_features
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from operator import itemgetter

def str2bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("yes", "true", "t", "1"):
        return True
    if str(value).lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def load_rest_config(path):
    if path is None:
        return {}
    with open(path, "r") as f:
        if str(path).endswith(".json"):
            return json.load(f)
        return yaml.load(f, Loader=yaml.Loader) or {}

def canonicalize_release_method(args):
    args.release_method = args.method
    if str(args.method).lower() == "rest":
        args.release_method = "REST"
        args.method = "rest_dqs" if args.setting == "DQS" else "rest_oqs"
    return args

def apply_release_config(args, config):
    mixed_corruption = args.image_corruption == "mixed" or args.text_corruption == "mixed"
    config_setting = config.get("setting")
    if args.setting is None:
        args.setting = str(config_setting or ("DQS" if mixed_corruption else "OQS")).upper()
    else:
        args.setting = args.setting.upper()
    if args.setting == "DQS" and args.image_corruption == "clean" and args.text_corruption == "clean":
        args.image_corruption = "mixed"
        args.text_corruption = "mixed"
    if args.gradient_decoupling is None:
        config_gd = config.get("gradient_decoupling")
        args.gradient_decoupling = str2bool(config_gd) if config_gd is not None else args.setting == "DQS"
    return canonicalize_release_method(args)

def generate_cirr_test_dicts(relative_test_dataset: CIRRDataset, adapt_model, index_features: torch.tensor,
                             index_names: List[str], txt_processors, save_memory =True) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compute test prediction dicts for CIRR dataset
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: test index features
    :param index_names: test index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: Top50 global and Top3 subset prediction for each query (reference_name, caption)
    """
    
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features[1]))

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    group_members = []
    reference_names = []
    distance = []
    captions_all = []
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss_total", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    
    if 'rest' in args.method:
        queue_list=[]
        max_queue_size=args.batch_size
        num_update_signal=10
        update_signal=True

    print('Test time adaptation...')
    adapt_model.eval()
    adapt_model.model.set_gallery_features(index_features[0].to(device), args.center)
    
    with torch.no_grad():
        for i, (batch_pairs_id, batch_reference_names, captions, batch_group_members) in enumerate(
            tqdm(relative_test_loader, desc="Processing", leave=False)
        ):

            if "rest" in args.method and i >= num_update_signal:
                    update_signal = False
            
            batch_group_members = np.array(batch_group_members).T.tolist()
            captions = [txt_processors["eval"](caption) for caption in captions]

            # Compute the predicted features
            if len(captions) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  
                
            if save_memory:
                reference_image_features = reference_image_features.to(device)
            
            if 'rest' in args.method:
                queue_list, batch_distance=adapt_model(reference_image_features, captions, args, metric_logger, queue_list, max_queue_size, update_signal)
            else:
                batch_distance=adapt_model(reference_image_features, captions, args, metric_logger)
            distance.append(batch_distance)
            captions_all += captions

            group_members.extend(batch_group_members)
            reference_names.extend(batch_reference_names)
            pairs_id.extend(batch_pairs_id)

    predicted_sim = torch.vstack(distance)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   

    pairid_to_predictions, pairid_to_group_predictions = utils.evaluation(predicted_sim, reference_names, captions_all, index_names,name_to_feat, group_members, pairs_id, rerank=False, blip_model=None
)

    return pairid_to_predictions, pairid_to_group_predictions

def save_or_load_features(preprocess, model, args):
    """
    根据 dress_type、image_corruption、text_corruption 自动创建特征文件夹，
    并保存/加载特征和名字。
    返回 index_features, index_names
    """
    # 构建特征保存目录，包含图像和文本扰动类型
    features_dir = os.path.join(
    args.feature_save_path,
    'img_' + args.image_corruption + ' + txt_' + args.text_corruption
    )
    os.makedirs(features_dir, exist_ok=True)

    features_path = os.path.join(features_dir, f"index_features.npz")
    names_path = os.path.join(features_dir, f"index_names.npy")

    if os.path.exists(features_path) and os.path.exists(names_path):
        print(f"Loading features for from npz...")
        data = np.load(features_path)
        index_features = (torch.tensor(data['gallery']), torch.tensor(data['all']))
        index_names = np.load(names_path, allow_pickle=True).tolist()
    else:
        print(f"Extracting and saving features ...")
        classic_test_dataset = CIRRDataset('test1', 'classic', preprocess, args)
        index_features, index_names = extract_index_blip_features(classic_test_dataset, model.model, save_memory=True, num_workers=args.num_workers)
        np.savez(features_path, gallery=index_features[0], all=index_features[1])
        np.save(names_path, np.array(index_names))
    return index_features, index_names

def main(args):
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)
    
    relative_test_dataset = CIRRDataset('test1', 'relative', preprocess, args)
    
    model, _, txt_processors = load_model_and_preprocess(name=args.blip_model_name, model_type=args.backbone, is_eval=False, device=device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    msg = model.load_state_dict(checkpoint[model.__class__.__name__], strict=False)
    print("Missing keys {}".format(msg.missing_keys))
    
    model = model.to(device)
    
    model = freeze_parameters(model)
    params = collect_params(model)[0]
    optimizer = set_optimizer(params, args=args)
    model = set_model(model, optimizer, args)
    
    start_time = time.time()
    
    index_features, index_names = save_or_load_features(preprocess, model, args)
    
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(relative_test_dataset, model, index_features, index_names, txt_processors, save_memory=True)
    
    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    if args.evaluation:
        submissions_folder_path = args.output_dir / 'evaluation' / ('img_' + args.image_corruption + ' + txt_' + args.text_corruption)
    else:
        submissions_folder_path = args.output_dir / args.release_method / ('img_' + args.image_corruption + ' + txt_' + args.text_corruption)
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    file_name = f'{args.blip_model_name}_2'

    print(f"Saving CIRR test predictions")
    with open(submissions_folder_path / f"recall_submission_{file_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"recall_subset_submission_{file_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument("--blip-model-name", default="blip2_cir_align_prompt", type=str)   
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--retrieval', type=str, default=None, help='Retrieval mode')
    parser.add_argument('--tta_steps', default=3, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--con_ratio', default=0.3, type=float)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--t', default=1, type=float)
    parser.add_argument('--method', default='REST')
    parser.add_argument('--setting', type=str.upper, choices=['OQS', 'DQS'], default=None)
    parser.add_argument('--gradient-decoupling', dest='gradient_decoupling', type=str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--rest-config', default=None, help='Optional YAML/JSON file with REST release settings.')
    parser.add_argument('--k_test', default=10, type=int)
    parser.add_argument('--center', default=64, type=int)
    parser.add_argument('--candidate_num', default=10, type=int)
    parser.add_argument("--model-path", type=str, default=str(Path(__file__).resolve().parents[2] / "weight" / "Composed_Image_Retrieval" / "SPRC" / "sprc_cirr.pt"), help="Path to the fine-tuned CLIP model")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory for release datasets. Defaults to REST_DATA_ROOT or ../data.")
    parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--lr", default=3e-5, type=float, help="Learning rate") #3e-5
    parser.add_argument("--image_corruption", default="clean", type=str) 
    parser.add_argument("--text_corruption", default="clean", type=str) 
    parser.add_argument("--feature_save_path", default=str(Path(tempfile.gettempdir()) / "rest_runs" / "cirr_features"), type=str)
    parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "rest_runs" / "cirr_predictions", help="Directory for CIRR prediction files.")
    parser.add_argument('--evaluation', default=0, type=int)

    args = parser.parse_args()
    args = apply_release_config(args, load_rest_config(args.rest_config))
    
    if args.evaluation:
        args.tta_steps = 1

    args = set_temperature(args)
    
    main(args)
