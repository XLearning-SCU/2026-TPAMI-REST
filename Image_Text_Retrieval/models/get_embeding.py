import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from lavis.models.base_model import (
    concat_all_gather,
)

@torch.no_grad()
def get_text_embeds(data_loader, model, device, args):
    num_length = len(data_loader.dataset.text)
    embed_dim = 256  # 假设输出维度固定为 256
    seq_len = 35     # 最大序列长度，跟 tokenizer 保持一致

    all_embeds = torch.zeros(num_length, embed_dim).to(device)
    all_input_ids = torch.zeros(num_length, seq_len, dtype=torch.long).to(device)
    all_attention_mask = torch.zeros(num_length, seq_len, dtype=torch.long).to(device)

    with autocast():
        for text, index in data_loader:
            text_input = model.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_tensors="pt"
            ).to(device)

            text_output = model.text_encoder.forward_text(text_input)
            text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

            # 存储 text embedding 和原始 input_ids/mask
            all_embeds[index] = text_embed
            all_input_ids[index] = text_input.input_ids
            all_attention_mask[index] = text_input.attention_mask

    if args.distributed:
        dist.all_reduce(all_embeds, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_input_ids, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_attention_mask, op=dist.ReduceOp.SUM)

    # 检查是否有空 embedding
    if torch.any(torch.all(all_embeds == 0, dim=1)):
        raise ValueError("There is at least one row in text_embeds that is all zeros.")

    return all_embeds, all_input_ids, all_attention_mask



def get_image_embeds(data_loader, model, device, args):
    num_length = len(data_loader.dataset.image)
    embed_dim = 256  # 假设输出维度固定为 256
    
    with autocast():
        all_embeds = torch.zeros(num_length, embed_dim).to(device)
        for image, index in data_loader:
            image = image.to(device)
            image_feat = model.visual_encoder.forward_features(image)
            image_embed = model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            all_embeds[index] = image_embed  # 支持 batch 索引张量

    if args.distributed:
        dist.all_reduce(all_embeds, op=dist.ReduceOp.SUM)

    if torch.any(torch.all(all_embeds == 0, dim=1)):
        raise ValueError("There is at least one row in image_embeds that is all zeros.")

    return all_embeds

def get_image_embeds_itm(data_loader, model, device, args):
    num_length = len(data_loader.dataset.image)
    embed_dim = 256  # 假设输出维度固定为 256
    
    image_feat_list = []  # 存储每次收集的特征
    
    with autocast():
        all_embeds = torch.zeros(num_length, embed_dim).to(device)
        for image, index in data_loader:
            image = image.to(device)
            image_feat = model.visual_encoder.forward_features(image)
            
            # 每次 forward 时就使用 concat_all_gather 进行跨卡数据收集，并转到 CPU
            gathered_feat = concat_all_gather(image_feat).cpu()
            image_feat_list.append(gathered_feat)
            
            image_embed = model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            all_embeds[index] = image_embed  # 支持 batch 索引张量

    # 如果是分布式训练，做全局同步
    if args.distributed:
        dist.all_reduce(all_embeds, op=dist.ReduceOp.SUM)

    # 检查是否有空的嵌入
    if torch.any(torch.all(all_embeds == 0, dim=1)):
        raise ValueError("There is at least one row in image_embeds that is all zeros.")
    
    image_feats= torch.cat(image_feat_list, dim=0)

    # 返回最终的嵌入和存储的特征列表
    return all_embeds, image_feats
