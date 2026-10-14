"""REST evaluation for image-text retrieval under online or diverse query shift."""
import argparse
import os
from tqdm import tqdm
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import tempfile
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import utils
from data import create_dataset, create_sampler, create_loader
from lavis.models.base_model import concat_all_gather
from lavis.models import load_model_and_preprocess
from models.set_model import *
from models.get_embeding import *

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
    config_setting = config.get("setting")
    if args.setting is None:
        args.setting = str(config_setting or ("DQS" if int(args.mixed_corruption) else "OQS")).upper()
    else:
        args.setting = args.setting.upper()

    if args.setting not in ("OQS", "DQS"):
        raise ValueError("setting should be either OQS or DQS.")
    if args.setting == "DQS":
        args.mixed_corruption = 1

    if args.gradient_decoupling is None:
        config_gd = config.get("gradient_decoupling")
        args.gradient_decoupling = str2bool(config_gd) if config_gd is not None else args.setting == "DQS"

    return canonicalize_release_method(args)

def test_time_tune(adapt_model, data_loader_text, data_loader_image, device, config, args):
    model_without_ddp=adapt_model.model.module
    print('Computing features for evaluation...')
    start_time = time.time()

    len_image=len(data_loader_image.dataset.image)
    len_text=len(data_loader_image.dataset.text)
    score_matrix_i2t = torch.full((len_image, len_text), -100.0).to(device)
    score_matrix_t2i = torch.full((len_text, len_image), -100.0).to(device)

    idx_list=[]
    
    adapt_model.eval()
    with torch.no_grad():
        if args.only_visual==True:
            all_text_embeds, _,_ =get_text_embeds(data_loader_text, model_without_ddp, device, args)
            model_without_ddp.set_gallery_features(all_text_embeds.detach(), args.center)
            if args.method == 'rest_dqs':
                adapt_model.teacher_model.module.set_gallery_features(all_text_embeds.detach(), args.center)
        else:
            all_image_embeds=get_image_embeds(data_loader_image, model_without_ddp, device, args)
            model_without_ddp.set_gallery_features(all_image_embeds.detach(), args.center)
            if args.method == 'rest_dqs':
                adapt_model.teacher_model.module.set_gallery_features(all_image_embeds.detach(), args.center)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_total', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    bs=config['batch_size']
    if 'rest' in args.method:
        queue_list=[]
        max_queue_size=config['batch_size']
        num_update_signal=10
        update_signal=True

    print('Test time adaptation...')
    with torch.no_grad():
        if args.only_visual:
            for i,(image, idx) in enumerate(tqdm(data_loader_image, desc='Processing', leave=False)):
                if ('rest' in args.method) and i>=num_update_signal:
                    update_signal=False
                image = image.to(device)
                idx_list.append(concat_all_gather(idx.to(device)))
                if 'rest' in args.method:
                    queue_list, sims_matrix=adapt_model(image, device, args, metric_logger, queue_list, max_queue_size, update_signal)
                else:
                    sims_matrix=adapt_model(image, device, args, metric_logger)

                start=i*bs
                end = min(score_matrix_i2t.size(0),(i+1)*bs)
                sims_matrix = sims_matrix[:end - start]
                score_matrix_i2t[start:end]=sims_matrix
        else:
            for i, (text, idx) in enumerate(tqdm(data_loader_text, desc='Processing', leave=False)):
                if ('rest' in args.method) and i>=num_update_signal:
                    update_signal=False
                idx_list.append(concat_all_gather(idx.to(device)))
                if 'rest' in args.method:
                    queue_list, sims_matrix=adapt_model(text, device, args, metric_logger, queue_list, max_queue_size, update_signal)
                else:
                    sims_matrix=adapt_model(text, device, args, metric_logger)

                start=i*bs
                end = min(score_matrix_t2i.size(0),(i+1)*bs)
                score_matrix_t2i[start:end]=sims_matrix

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Test time adaptation time {}'.format(total_time_str))

    idx_tensor = torch.cat(idx_list)
    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(), idx_tensor.cpu().numpy()


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    
    args.backbone=config['backbone']

    test_dataset_text, test_dataset_image = create_dataset('retrieval_%s'%config['dataset'], config)  
          
    if args.distributed and args.retrieval is not None:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()           
        #shuffle! 
        if args.retrieval=='t2i':
            samplers = create_sampler([test_dataset_text], [True], num_tasks, global_rank) + create_sampler([test_dataset_image], [False], num_tasks, global_rank)
        elif args.retrieval=='i2t':
            samplers = create_sampler([test_dataset_text], [False], num_tasks, global_rank) + create_sampler([test_dataset_image], [True], num_tasks, global_rank) 
        batch_size=int(config['batch_size']/num_tasks)
    else:
        samplers = [None,None]
        batch_size=int(config['batch_size'])
    
    test_loader_text, test_loader_image = create_loader([test_dataset_text, test_dataset_image], samplers,
                                    batch_size=[batch_size,              
                                    batch_size],
                                    num_workers=[1, 1],
                                    is_trains=[False,False], 
                                    collate_fns=[None, None],config=config)   

    #### Model #### 
    print("Creating model")

    config['weight_decay'] = 1e-3
    
    model, vis_processors, txt_processors = load_model_and_preprocess(name=config['baseline'], model_type=config['dataset'], is_eval=False, config=args.config, device=device)
    
    model = model.to(device)
    if args.retrieval is not None:
        args.only_visual = True if args.retrieval == 'i2t' else False
        model = freeze_parameters(model, only_visual=args.only_visual)
    
    if args.distributed and args.retrieval is not None:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        params = collect_params(model.module, only_visual=args.only_visual)[0]
        optimizer = set_optimizer(params, config, args=args)
    else:
        optimizer = None

    if args.retrieval is not None:
        model = set_model(model, optimizer, args)

    print("Start evaluation")
    start_time = time.time()    
    if args.retrieval is not None:
        score_test_i2t, score_test_t2i, idx_trans = test_time_tune(model, test_loader_text, test_loader_image, device, config, args)
        test_result = utils.itm_eval(score_test_i2t, score_test_t2i, args, test_loader_image, idx_trans)
    else:
        score_test_i2t, score_test_t2i = model.compute_sim_matrix(test_loader_text, test_loader_image, device, args)
        test_result = utils.itm_eval(score_test_i2t, score_test_t2i, args, test_loader_image)
        
    print(f"config['image_root']={config['image_root']}")
    print(f"config['ann_root']={config['ann_root']}")
    print(f"config['dataset']={config['dataset']}")
    print(f"config['backbone']={config['backbone']}")
    print(f"config['temperature']={args.temperature}")
    print(f"config['candidate_num']={args.candidate_num}")
    print(f"test_result is: {test_result}")
        
    log_stats = {**{f'test_{k}': v for k, v in test_result.items()},                  
                }
    
    print("Evaluation Results:")
    for k, v in log_stats.items():
        print(f"{k}: {v}")
    
    with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
        f.write(json.dumps(log_stats) + "\n")    

    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./lavis/configs/models/blip_retrieval_flickr.yaml')
    parser.add_argument('--model-path', default=None, help='Override the checkpoint path in the task config.')
    parser.add_argument('--output_dir', default=str(Path(tempfile.gettempdir()) / 'rest_runs' / 'Image_Text_Retrieval'))
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--retrieval', type=str, default=None, help='Retrieval mode')
    parser.add_argument('--tta_steps', default=3, type=int)
    parser.add_argument('--con_ratio', default=0.3, type=float)
    parser.add_argument('--temperature', default=0.02, type=float)
    parser.add_argument('--t', default=1, type=float)
    parser.add_argument('--method', default='REST')
    parser.add_argument('--setting', type=str.upper, choices=['OQS', 'DQS'], default=None)
    parser.add_argument('--gradient-decoupling', dest='gradient_decoupling', type=str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--rest-config', default=None, help='Optional YAML/JSON file with REST release settings.')
    parser.add_argument('--k_test', default=128, type=int)
    parser.add_argument('--center', default=64, type=int)
    parser.add_argument('--mixed_corruption', default=0, type=int)
    parser.add_argument('--candidate_num', default=10, type=int)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if args.model_path:
        config['model']['finetuned'] = args.model_path
    else:
        configured_weight = Path(config['model']['finetuned'])
        published_weight = (Path(__file__).resolve().parents[2] / 'weight' /
                            'Image_Text_Retrieval' / 'BLIP' / configured_weight.name)
        if not configured_weight.is_file() and published_weight.is_file():
            config['model']['finetuned'] = str(published_weight)
    data_root = os.environ.get('REST_DATA_ROOT')
    if data_root:
        for key in ('image_root', 'ann_root'):
            value = config.get(key)
            if value and value.startswith('../../data/Image_Text_Retrieval/'):
                task_root = Path(data_root)
                if task_root.name != 'Image_Text_Retrieval':
                    task_root = task_root / 'Image_Text_Retrieval'
                config[key] = str(task_root / value.split('Image_Text_Retrieval/', 1)[1])
    release_config = {**config, **load_rest_config(args.rest_config)}
    args = apply_release_config(args, release_config)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    args, config = set_temperature(args, config)

    if args.model_path:
        with tempfile.TemporaryDirectory(prefix='rest_config_') as temporary_dir:
            effective_config = Path(temporary_dir) / 'config.yaml'
            with effective_config.open('w') as stream:
                yaml.dump(config, stream)
            args.config = str(effective_config)
            main(args, config)
    else:
        main(args, config)
