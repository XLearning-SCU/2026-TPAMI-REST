import torch
from models.tta_baselines.tent import Tent
from models.tta_baselines.sar import SAR, SAM
from models.tta_baselines.eata import EATA
from models.tta_baselines.read import READ
from models.rest_methods.rest_dqs import REST_DQS
from models.rest_methods.rest import REST
from models.tta_baselines.come import COME
from models.tta_baselines.pl import PseudoLabel
from torch import nn

def set_optimizer(params, config, args):
    if args.method=='sar':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(params=params, base_optimizer=base_optimizer, lr=config['init_lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(params=params, lr=config['init_lr'], weight_decay=config['weight_decay']) 
    return optimizer

def set_model(model, optimizer, args):
    if args.method=='tent':
        model=Tent(model, optimizer, steps=args.tta_steps)
    elif args.method=='pl':
        model=PseudoLabel(model, optimizer, steps=args.tta_steps, threshold=0.5, clf_coeff=0.1)
    elif args.method=='sar':
        model=SAR(model, optimizer, steps=args.tta_steps, margin_e0=0.40)
    elif args.method=='eata':
        model=EATA(model, optimizer, steps=args.tta_steps, fishers=None, e_margin=0.40)
    elif args.method=='read':
        model=READ(model, optimizer, steps=args.tta_steps)
    elif args.method=='come':
        model=COME(model, optimizer, steps=args.tta_steps)
    elif args.method=='rest_oqs':
        model=REST(model, optimizer, steps=args.tta_steps)
    elif args.method=='rest_dqs':
        model=REST_DQS(model, optimizer, steps=args.tta_steps)
    else:
        assert False, "Not correct model name."

    return model

def freeze_parameters(model, only_visual):
    model.train()
    model.requires_grad_(False)
    if only_visual:
        print("only_visual")
        for name, param in model.visual_encoder.named_parameters():
            if ('norm' in name) or ('Norm' in name):
                param.requires_grad_(True)
    else:
        print("only_text")
        for name, param in model.text_encoder.named_parameters():
            if ('norm' in name) or ('Norm' in name): 
                param.requires_grad_(True)
    return model

def collect_params(model, only_visual):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    if only_visual:
        for nm, m in model.visual_encoder.named_modules():
            if isinstance(m, (nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
    else:
        for nm, m in model.text_encoder.named_modules():
            if isinstance(m, (nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

    return params, names


def set_temperature(args, config):
    if args.method in ('rest_oqs', 'rest_dqs'): 
        return args, config 
    
    if args.retrieval == 'i2t' and args.mixed_corruption == 0: 
        config['init_lr']=3e-4

    return args, config


def convert_weights_to_fp16(model: nn.Module):
    """Apply the same FP16 conversion used by the original BLIP entrypoint."""
    def _convert_weights_to_fp16(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            if layer.weight is not None:
                layer.weight.data = layer.weight.data.half()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.half()

        if isinstance(layer, nn.MultiheadAttention):
            for name in [
                *[f"{side}_proj_weight" for side in ("in", "q", "k", "v")],
                "in_proj_bias", "bias_k", "bias_v",
            ]:
                value = getattr(layer, name, None)
                if isinstance(value, torch.Tensor):
                    value.data = value.data.half()

        for name in ("text_projection", "proj"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                value.data = value.data.half()
            elif isinstance(value, nn.Module):
                if hasattr(value, "weight") and isinstance(value.weight, torch.Tensor):
                    value.weight.data = value.weight.data.half()
                if hasattr(value, "bias") and isinstance(value.bias, torch.Tensor) and value.bias is not None:
                    value.bias.data = value.bias.data.half()

    model.apply(_convert_weights_to_fp16)
