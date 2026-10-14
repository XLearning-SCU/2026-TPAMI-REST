import torch
from models.tta_baselines.tent import Tent
from models.tta_baselines.sar import SAR, SAM
from models.tta_baselines.eata import EATA
from models.tta_baselines.read import READ
from models.tta_baselines.shot import SHOT
from models.rest_methods.rest import REST
from models.tta_baselines.come import COME
from models.tta_baselines.pl import PseudoLabel
from models.tta_baselines.tsa import TSA
from models.rest_methods.rest_dqs import REST_DQS

def set_optimizer(params, args):
    if args.method == 'sar':
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params=params,
            base_optimizer=base_optimizer,
            lr=args.lr,
            weight_decay=0.
        )
    else:
        optimizer = torch.optim.Adam(
            params=[{'params': params, 'lr': args.lr}],
            weight_decay=0.,
            betas=(0.9, 0.999)
        )
    return optimizer


def set_model(model, optimizer, args):
    if args.method=='tent':
        model=Tent(model, optimizer, steps=args.tta_steps)
    elif args.method=='pl':
        model=PseudoLabel(model, optimizer, steps=args.tta_steps, threshold=0.5, clf_coeff=0.1)
    elif args.method=='sar':
        model=SAR(model, optimizer, steps=args.tta_steps, margin_e0=0.40)
    elif args.method=='shot':
        model=SHOT(model, optimizer, steps=args.tta_steps, threshold=0.9, clf_coeff=0.1)
    elif args.method=='eata':
        model=EATA(model, optimizer, steps=args.tta_steps, fishers=None, e_margin=0.40)
    elif args.method=='read':
        model=READ(model, optimizer, steps=args.tta_steps)
    elif args.method=='come':
        model=COME(model, optimizer, steps=args.tta_steps)
    elif args.method=='tsa':
        model=TSA(model, optimizer, steps=args.tta_steps)
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

    for name, param in model.named_parameters():
        if only_visual:
            if (
                name.startswith("blocks_v") and ("norm1" in name or "norm2" in name)
                or name.startswith("blocks_u") and ("norm1_v" in name or "norm2_v" in name)
                or name.startswith("norm_v")
            ):
                param.requires_grad = True
        else:
            if (
                name.startswith("blocks_a") and ("norm1" in name or "norm2" in name)
                or name.startswith("blocks_u") and ("norm1_a" in name or "norm2_a" in name)
                or name.startswith("norm_a")
            ):
                param.requires_grad = True
    return model

def collect_params(model, only_visual):
    params = []
    names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if only_visual:
            if (
                name.startswith("blocks_v") and ("norm1" in name or "norm2" in name)
                or name.startswith("blocks_u") and ("norm1_v" in name or "norm2_v" in name)
                or name.startswith("norm_v")
            ):
                params.append(param)
                names.append(name)
        else:
            if (
                name.startswith("blocks_a") and ("norm1" in name or "norm2" in name)
                or name.startswith("blocks_u") and ("norm1_a" in name or "norm2_a" in name)
                or name.startswith("norm_a")
            ):
                params.append(param)
                names.append(name)
    return params, names

def set_temperature(args):
    if args.method in ('rest_oqs', 'rest_dqs'): 
        return args 

    return args
