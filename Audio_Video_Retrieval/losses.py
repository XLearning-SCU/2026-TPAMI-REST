import torch
import torch.nn.functional as F

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1))

def robust_entropy_minimization_loss(outputs, entropy_queue, eps=1e-3):
    entropys=softmax_entropy(outputs).sum(1)
    entropy_threshold=entropy_queue.max()
    weight=torch.clamp(torch.tensor(1.0) - entropys.clone().detach() /(entropy_threshold + eps), min=0) #0.1,2
    loss=entropys.mul(weight)[entropys<=entropy_threshold]
    return loss.mean(0)

def robust_entropy_minimization_loss_with_weight(outputs, entropy_queue, eps=1e-3):
    entropys = softmax_entropy(outputs).sum(1)  # [B]
    entropy_threshold = entropy_queue.max()

    # 原始 weight 计算
    raw_weight = 1.0 - entropys.clone().detach() / (entropy_threshold + eps)
    raw_weight = torch.clamp(raw_weight, min=0.0)  # [B]

    # mask 出满足条件的位置
    mask = entropys <= entropy_threshold  # [B], bool
    masked_weight = raw_weight * mask.float()  # 其他位置为0

    # 用 masked_weight 算 loss，跟原来逻辑完全一样
    loss = (entropys * masked_weight)[mask].mean(0)

    return loss, masked_weight


def compute_query_gallery_gap(all_image_embeds,all_text_embeds):
    all_image_embeds=F.normalize(all_image_embeds)
    all_text_embeds=F.normalize(all_text_embeds)
    image_embed=all_image_embeds.mean(dim=0)
    text_embed=all_text_embeds.mean(dim=0)
    query_gallery_shift = image_embed - text_embed
    query_gallery_gap = torch.norm(query_gallery_shift, p=2)
    return query_gallery_gap

@torch.no_grad()
def get_current_value(queue_list):
    _, modality_1_embeds, modality_2_embeds, entropys = zip(*queue_list)
    feat_1_queue = torch.stack(modality_1_embeds)
    feat_2_queue = torch.stack(modality_2_embeds)
    current_margin=compute_query_gallery_gap(feat_1_queue, feat_2_queue)
    entropys_queue=torch.stack(entropys)
    return current_margin, entropys_queue


@torch.no_grad()
def update_queue(modality_1_feat, modality_2_feat, queue_list, con_ratio, max_queue_size, args):
    num_to_select = int(con_ratio * modality_1_feat.size(0))

    sample_gap=torch.norm(modality_1_feat - modality_2_feat, p=2, dim=1)
    modality_1_center=modality_1_feat.mean(0)
    modality_2_center=modality_2_feat.mean(0)#all_modality_2_feat.mean(0)#modality_2_feat.mean(0)
    sample_1_diversity=torch.norm(modality_1_feat - modality_1_center, p=2, dim=1)
    sample_2_diversity=torch.norm(modality_2_feat - modality_2_center, p=2, dim=1)
    indictor=2*sample_gap-sample_1_diversity-sample_2_diversity

    entropys=softmax_entropy(modality_1_feat@modality_2_feat.t()/args.temperature).sum(1)

    sorted_indices = torch.argsort(indictor)[:num_to_select]

    for i in sorted_indices:
        indictor_item = indictor[i].detach().item()
        modality_1_feat_item = modality_1_feat[i].detach()
        modality_2_feat_item = modality_2_feat[i].detach()
        entropys_item=entropys[i].detach()
        queue_list.append((indictor_item,modality_1_feat_item, modality_2_feat_item,entropys_item))

    # Min Stack Update
    if len(queue_list) >= max_queue_size:
        queue_list = sorted(queue_list, key=lambda x: x[0], reverse=False)
        queue_list = queue_list[:max_queue_size]

    return queue_list

@torch.no_grad()
def update_queue_new(modality_1_feat, modality_2_feat, sim_inter, queue_list, max_queue_size, args):
    num_to_select = int(args.con_ratio * modality_1_feat.size(0))

    sample_gap=torch.norm(modality_1_feat - modality_2_feat, p=2, dim=1)
    modality_1_center=modality_1_feat.mean(0)
    modality_2_center=modality_2_feat.mean(0) #all_modality_2_feat.mean(0)#modality_2_feat.mean(0)
    sample_1_diversity=torch.norm(modality_1_feat - modality_1_center, p=2, dim=1)
    sample_2_diversity=torch.norm(modality_2_feat - modality_2_center, p=2, dim=1)

    indictor=2*sample_gap-sample_1_diversity-sample_2_diversity

    entropys=softmax_entropy(sim_inter).sum(1)

    sorted_indices = torch.argsort(indictor)[:num_to_select]

    for i in sorted_indices:
        indictor_item = indictor[i].detach().item()
        modality_1_feat_item = modality_1_feat[i].detach()
        modality_2_feat_item = modality_2_feat[i].detach()
        entropys_item=entropys[i].detach()
        queue_list.append((indictor_item,modality_1_feat_item, modality_2_feat_item,entropys_item))

    # Min Stack Update
    if len(queue_list) >= max_queue_size:
        queue_list = sorted(queue_list, key=lambda x: x[0], reverse=False)
        queue_list = queue_list[:max_queue_size]

    return queue_list

def query_uniformity_loss(x):
    center=x.mean(0).detach()
    distances = torch.norm(x - center, dim=1) 
    loss = (torch.exp(-distances)).mean()
    return loss

def get_REM_output(query_feat, gallery_feat_all, cluster, k=10):
    n, D = query_feat.shape
    topk_all = query_feat @ gallery_feat_all.t()  # [n, N]
    topk_indices = topk_all.topk(k=k, dim=1).indices  # [n, k]
    pos_indices = topk_indices[:, :1]  # [n, 1]
    self_mask = torch.eye(n, dtype=torch.bool, device=query_feat.device)  # [n, n]
    topk_expanded = topk_indices.unsqueeze(0).expand(n, -1, -1)  # [n, n, k]
    neg_indices = topk_expanded[~self_mask].reshape(n, (n - 1) * k)  # [n, (n-1)*k]
    final_indices = torch.cat([pos_indices, neg_indices], dim=1)  # [n, 1 + (n-1)*k]
    selected_gallery_feats = gallery_feat_all[final_indices]  # [n, total, D]
    
    C = cluster.size(0)  # 总 cluster 数
    n = query_feat.size(0)
    all_indices = torch.randperm(C, device=query_feat.device)[:k]  # [k]
    cluster_neg_feats = cluster[all_indices]  # [k, D]
    cluster_gallery_feat = cluster_neg_feats.unsqueeze(0).expand(n, -1, -1)  # [n, k, D]

    # Step 3: 合并构成最终 gallery
    final_gallery_feats = torch.cat([selected_gallery_feats, cluster_gallery_feat], dim=1)  # [n, 5n, D]

    # Step 4: 相似度计算
    outputs = torch.bmm(
        query_feat.unsqueeze(1),              # [n, 1, D]
        final_gallery_feats.transpose(1, 2)            # [n, D, 5n]
    ).squeeze(1)  # [n, 5n]
    
    return outputs

def get_REM_output_mixed(query_feat, gallery_feat_all, cluster, k=10):
    n, D = query_feat.shape
    topk_all = query_feat @ gallery_feat_all.t()  # [n, N]
    topk_indices = topk_all.topk(k=k, dim=1).indices  # [n, k]
    pos_indices = topk_indices[:, :1]  # [n, 1]
    self_mask = torch.eye(n, dtype=torch.bool, device=query_feat.device)  # [n, n]
    topk_expanded = topk_indices.unsqueeze(0).expand(n, -1, -1)  # [n, n, k]
    neg_indices = topk_expanded[~self_mask].reshape(n, (n - 1) * k)  # [n, (n-1)*k]
    final_indices = torch.cat([pos_indices, neg_indices], dim=1)  # [n, 1 + (n-1)*k]
    selected_gallery_feats = gallery_feat_all[final_indices]  # [n, total, D]
    
    C = cluster.size(0)  # 总 cluster 数
    n = query_feat.size(0)
    all_indices = torch.randperm(C, device=query_feat.device)[:k]  # [k]
    cluster_neg_feats = cluster[all_indices]  # [k, D]
    cluster_gallery_feat = cluster_neg_feats.unsqueeze(0).expand(n, -1, -1)  # [n, k, D]

    # Step 3: 合并构成最终 gallery
    final_gallery_feats = torch.cat([selected_gallery_feats, cluster_gallery_feat], dim=1)  # [n, 5n, D]

    # Step 4: 相似度计算
    outputs = torch.bmm(
        query_feat.unsqueeze(1),              # [n, 1, D]
        final_gallery_feats.transpose(1, 2)            # [n, D, 5n]
    ).squeeze(1)  # [n, 5n]
    
    return outputs, final_gallery_feats

def robust_hard_mining_loss(sim_matrix, weight, k = 100):
    topk_values, _ = sim_matrix.sort(dim=1, descending=True)
    topk_prob = F.softmax(topk_values, dim=1)  # [B, 100] 使用 softmax 获得概率
    top1_sim = topk_values[:, 0]  # [B]
    neg_idx = torch.multinomial(topk_prob[:, 1:], 1).squeeze(1)  # 从 softmax 概率中采样，排除 top1
    neg_sim = topk_values[torch.arange(topk_values.size(0)), neg_idx + 1] 
    top1_sim = torch.clamp(top1_sim, min=0.0)
    neg_sim = torch.clamp(neg_sim, min=0.0)
    top1_vec = torch.stack([1 - top1_sim, top1_sim], dim=1)  # [B, 2]
    neg_vec = torch.stack([1 - neg_sim, neg_sim], dim=1)  # [B, 2]
    logits = torch.cat([top1_vec, neg_vec], dim=0)  # [2B, 2]
    target = torch.cat([
        torch.ones(top1_vec.size(0), dtype=torch.long, device=top1_vec.device),
        torch.zeros(neg_vec.size(0), dtype=torch.long, device=neg_vec.device)
    ], dim=0)  # [2B]

    loss_all = F.cross_entropy(logits, target, reduction='none')
    sample_weight = torch.cat([weight, weight], dim=0)  # shape: [64]
    loss = (loss_all * sample_weight).mean()
    return loss

