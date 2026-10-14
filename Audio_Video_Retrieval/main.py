"""REST evaluation for audio-video retrieval under online or diverse query shift."""
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import ruamel.yaml as yaml
import time
import datetime
import torch
import torch.backends.cudnn as cudnn
import dataloader as dataloader
import models
from torch.utils.data import DataLoader
from models.set_model import set_optimizer, set_model, set_temperature, freeze_parameters, collect_params
from models.cav_mae import get_audio_embeds, get_video_embeds, compute_sim_matrix
import utils

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

    if args.setting == "DQS":
        args.mixed_corruption = 1
    if args.gradient_decoupling is None:
        config_gd = config.get("gradient_decoupling")
        args.gradient_decoupling = str2bool(config_gd) if config_gd is not None else args.setting == "DQS"

    return canonicalize_release_method(args)

def test_time_tune(adapt_model, data_loader, device, args):
    print('Computing features for evaluation...')
    start_time = time.time()

    num_samples = data_loader.dataset.num_samples
    bs = args.batch_size

    score_matrix_v2a = torch.full((num_samples, num_samples), -100.0, device=device)
    score_matrix_a2v = torch.full((num_samples, num_samples), -100.0, device=device)

    adapt_model.eval()

    with torch.no_grad():
        if args.only_visual:
            all_audio_embeds = get_audio_embeds(data_loader, adapt_model.model, device)
            adapt_model.model.set_gallery_features(all_audio_embeds.detach(), args.center)
            if args.method == 'rest_dqs':
                adapt_model.teacher_model.set_gallery_features(all_audio_embeds.detach(), args.center)
        else:
            all_video_embeds = get_video_embeds(data_loader, adapt_model.model, device)
            adapt_model.model.set_gallery_features(all_video_embeds.detach(), args.center)
            if args.method == 'rest_dqs':
                adapt_model.teacher_model.set_gallery_features(all_video_embeds.detach(), args.center)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss_total", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))

    if 'rest' in args.method:
        queue_list = []
        max_queue_size = args.batch_size
        num_update_signal = 10
        update_signal = True

    print('Test time adaptation...')
    with torch.no_grad():
        if args.only_visual:
            for i, (a_input, v_input) in enumerate(tqdm(data_loader, desc='Processing', leave=False)):
                if ('rest' in args.method) and i >= num_update_signal:
                    update_signal = False
                v_input = v_input.to(device)

                if 'rest' in args.method:
                    queue_list, sims_matrix = adapt_model(v_input, device, args, metric_logger, queue_list, max_queue_size, update_signal)
                else:
                    sims_matrix = adapt_model(v_input, device, args, metric_logger)

                start = i * bs
                end = min(score_matrix_v2a.size(0), (i + 1) * bs)
                if num_samples % bs != 0 and end == num_samples and sims_matrix.shape[0] > end - start:
                    sims_matrix = sims_matrix[:end - start]
                score_matrix_v2a[start:end] = sims_matrix
        else:
            for i, (a_input, v_input) in enumerate(tqdm(data_loader, desc='Processing', leave=False)):
                if ('rest' in args.method) and i >= num_update_signal:
                    update_signal = False
                a_input = a_input.to(device)

                if 'rest' in args.method:
                    queue_list, sims_matrix = adapt_model(a_input, device, args, metric_logger, queue_list, max_queue_size, update_signal)
                else:
                    sims_matrix = adapt_model(a_input, device, args, metric_logger)

                start = i * bs
                end = min(score_matrix_a2v.size(0), (i + 1) * bs)
                if num_samples % bs != 0 and end == num_samples and sims_matrix.shape[0] > end - start:
                    sims_matrix = sims_matrix[:end - start]
                score_matrix_a2v[start:end] = sims_matrix

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Test time adaptation time {}'.format(total_time_str))

    return score_matrix_v2a.cpu().numpy(), score_matrix_a2v.cpu().numpy()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating retrieval dataset...")

    im_res = 224
    val_audio_conf = {
        'num_mel_bins': 128,
        'target_length': 1024,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': 'audioset',
        'mode': 'eval',
        'mean': -5.081,
        'std': 4.4849,
        'noise': False,
        'im_res': im_res
    }

    val_dataset = dataloader.AudiosetDataset(args.json_root, audio_conf=val_audio_conf)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("Loading model...")

    model = models.CAVMAEFT(modality_specific_depth=11)
    checkpoint = torch.load(args.model_path, map_location=device)
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    
    if args.retrieval is not None:
        args.only_visual = True if args.retrieval == 'v2a' else False
        model = freeze_parameters(model, only_visual=args.only_visual)
        params = collect_params(model, only_visual=args.only_visual)[0]
        optimizer = set_optimizer(params, args=args)
        model = set_model(model, optimizer, args)
    
    print("Start evaluation")
    t0 = time.time()
    if args.retrieval is not None:
        score_v2a, score_a2v = test_time_tune(model, val_loader, device, args)
        test_result = utils.itm_eval(score_v2a, score_a2v)
    else:
        score_v2a, score_a2v = compute_sim_matrix(model, val_loader, device)
        test_result = utils.itm_eval(score_v2a, score_a2v)
        
    if args.retrieval == "v2a":
        print("Test Result:", {k: test_result[k] for k in ['v2a_r1', 'v2a_r5', 'v2a_r10', 'v2a_r_mean', 'v2a_map']})
    elif args.retrieval == "a2v":
        print("Test Result:", {k: test_result[k] for k in ['a2v_r1', 'a2v_r5', 'a2v_r10', 'a2v_r_mean', 'a2v_map']})
    else:
        print("Test Result:", test_result)

        
    print("Evaluation time:", str(datetime.timedelta(seconds=int(time.time()-t0))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument("--json_root", type=str, default='config/OQS/v2a/video_noise_blur_digital.json', help="validation data json")
    parser.add_argument("--model", type=str, default='cav-mae-ft', help="the model used")
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, metavar='LR', help='initial learning rate')     
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
    parser.add_argument('--batch_size', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--num_workers', '--num_workers', default=8, type=int, metavar='N', help='mini-batch size')
    parser.add_argument("--model_path", type=str, default=str(Path(__file__).resolve().parents[2] / 'weight' / 'Audio_Video_Retrieval' / 'cav-mae' / 'audio_model.21.pth'), help="pretrained model path")
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
    parser.add_argument('--center', default=32, type=int)
    parser.add_argument('--candidate_num', default=10, type=int)
    parser.add_argument('--mixed_corruption', default=0, type=int)
    
    args = parser.parse_args()
    args = apply_release_config(args, load_rest_config(args.rest_config))
    args = set_temperature(args)
    
    main(args)
