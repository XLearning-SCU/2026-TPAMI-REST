"""REST evaluation for FashionIQ composed image retrieval."""
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
from data_utils import targetpad_transform, FashionIQDataset
from utils import device, extract_index_blip_features, collate_fn
from typing import List, Tuple
from torch.utils.data import DataLoader
from operator import itemgetter
from statistics import mean

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

def generate_fiq_val_predictions(adapt_model, relative_val_dataset: FashionIQDataset,
                                 index_names: List[str], index_features, txt_processors, save_memory =True) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features[1]))

    # Initialize pairs_id, predicted_features, group_members and reference_names
    target_names = []
    reference_names_all = []
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
    if save_memory:
        adapt_model.model.set_gallery_features(index_features[0].to(device).detach(), args.center)
    
    for i, (reference_names, batch_target_names, captions) in enumerate(
        tqdm(relative_val_loader, desc="Processing", leave=False)
    ):

        if "rest" in args.method and i >= num_update_signal:
                update_signal = False
        
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        input_captions = [txt_processors["eval"](caption) for caption in input_captions]
        
        if len(input_captions) == 1:
            reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
        else:
            reference_image_features = torch.stack(itemgetter(*reference_names)(
                name_to_feat))  
        
        if save_memory:
            reference_image_features = reference_image_features.to(device)
        
        if 'rest' in args.method:
            queue_list, batch_distance=adapt_model(reference_image_features, input_captions, args, metric_logger, queue_list, max_queue_size, update_signal)
        else:
            batch_distance=adapt_model(reference_image_features, input_captions, args, metric_logger)
        distance.append(batch_distance)
        captions_all += input_captions
        
        target_names.extend(batch_target_names)
        reference_names_all.extend(reference_names)

    distance = torch.vstack(distance)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    
    return distance, target_names, reference_names_all, captions_all

def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, model, index_features: torch.tensor,
                            index_names: List[str], txt_processors) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """

    # Generate predictions
    pred_sim, target_names, reference_names, captions_all = generate_fiq_val_predictions(model, relative_val_dataset, index_names, index_features, txt_processors)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Compute the distances and sort the results
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50

def save_or_load_features(dress_type, preprocess, model, args):
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

    features_path = os.path.join(features_dir, f"{dress_type}_index_features.npz")
    names_path = os.path.join(features_dir, f"{dress_type}_index_names.npy")

    if os.path.exists(features_path) and os.path.exists(names_path):
        print(f"Loading features for {dress_type} from npz...")
        data = np.load(features_path)
        index_features = (torch.tensor(data['gallery']), torch.tensor(data['all']))
        index_names = np.load(names_path, allow_pickle=True).tolist()
    else:
        print(f"Extracting and saving features for {dress_type} ...")
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, args)
        index_features, index_names = extract_index_blip_features(classic_val_dataset, model.model, save_memory=True, num_workers=args.num_workers)
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

    idx_to_dress_mapping = {}
    for idx, dress_type in enumerate(args.val_dress_types):
        idx_to_dress_mapping[idx] = dress_type

    start_time = time.time()
    recalls_at10 = []
    recalls_at50 = []

    with torch.no_grad():
        for idx, dress_type in idx_to_dress_mapping.items():
            relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, args)
            
            model, _, txt_processors = load_model_and_preprocess(
                name=args.blip_model_name, model_type=args.backbone, is_eval=False, device=device
            )

            checkpoint = torch.load(args.model_path, map_location=device)
            msg = model.load_state_dict(checkpoint[model.__class__.__name__], strict=False)
            print("Missing keys {}".format(msg.missing_keys))
            
            model = model.to(device)
            
            model = freeze_parameters(model)
            params = collect_params(model)[0]
            optimizer = set_optimizer(params, args=args)
            model = set_model(model, optimizer, args)
            
            index_features, index_names = save_or_load_features(dress_type, preprocess, model, args)

            recall_at10, recall_at50 = compute_fiq_val_metrics(
                relative_val_dataset, model, index_features, index_names, txt_processors
            )
            recalls_at10.append(recall_at10)
            recalls_at50.append(recall_at50)
            
            print(f"[{dress_type}] Recall@10: {recall_at10:.2f}%, Recall@50: {recall_at50:.2f}%")  

            del model, optimizer, checkpoint, index_features, index_names
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    results_dict = {}
    for i in range(len(recalls_at10)):
        results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
        results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
    results_dict.update({
        f'average_recall_at10': mean(recalls_at10),
        f'average_recall_at50': mean(recalls_at50),
        f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
    })

    print(json.dumps(results_dict, indent=4))


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
    parser.add_argument('--temperature', default=0.02, type=float)
    parser.add_argument('--t', default=1, type=float)
    parser.add_argument('--method', default='REST')
    parser.add_argument('--setting', type=str.upper, choices=['OQS', 'DQS'], default=None)
    parser.add_argument('--gradient-decoupling', dest='gradient_decoupling', type=str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--rest-config', default=None, help='Optional YAML/JSON file with REST release settings.')
    parser.add_argument('--k_test', default=128, type=int)
    parser.add_argument('--center', default=64, type=int)
    parser.add_argument("--model-path", type=str, default=str(Path(__file__).resolve().parents[2] / "weight" / "Composed_Image_Retrieval" / "SPRC" / "sprc_fiq.pt"), help="Path to the fine-tuned CLIP model")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory for release datasets. Defaults to REST_DATA_ROOT or ../data.")
    parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate")
    parser.add_argument("--val_dress_types", nargs='+', default=['dress', 'toptee', 'shirt'], type=str, help="Validation dress types")
    parser.add_argument("--image_corruption", default="clean", type=str) 
    parser.add_argument("--text_corruption", default="clean", type=str) 
    parser.add_argument("--feature_save_path", default=str(Path(tempfile.gettempdir()) / "rest_runs" / "fiq_features"), type=str)
    parser.add_argument('--evaluation', default=0, type=int)
    parser.add_argument('--candidate_num', default=10, type=int)
    
    args = parser.parse_args()
    args = apply_release_config(args, load_rest_config(args.rest_config))
    
    if args.evaluation:
        args.tta_steps = 1
    
    args = set_temperature(args)
    
    main(args)
