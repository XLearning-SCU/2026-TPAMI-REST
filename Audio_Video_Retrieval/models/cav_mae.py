# -*- coding: utf-8 -*-
# @Time    : 3/11/23 4:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : cav_mae.py

import os
os.environ['TORCH_HOME'] = './pretrained_models'
import random
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
from .pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F
from tqdm import tqdm
import faiss
from losses import *

def get_audio_embeds(val_loader, audio_model, device):
    A_a_feat = []
    audio_model.eval()
    with torch.no_grad():
        for a_input, v_input in tqdm(val_loader, desc="Extracting audio features"):
            audio_input = a_input.to(device)
            with torch.cuda.amp.autocast():
                audio_output = audio_model.forward_feat(a=audio_input, mode='a')
            audio_output = audio_output
            A_a_feat.append(audio_output)
    A_a_feat = torch.cat(A_a_feat)
    # 检查有无全零特征
    if torch.any(torch.all(A_a_feat == 0, dim=1)):
        raise ValueError("There is at least one row in audio_embeds that is all zeros.")
    return A_a_feat

def get_video_embeds(val_loader, audio_model, device):
    A_v_feat = []
    audio_model.eval()
    with torch.no_grad():
        for a_input, v_input in tqdm(val_loader, desc="Extracting video features"):
            video_input = v_input.to(device)
            with torch.cuda.amp.autocast():
                video_output = audio_model.forward_feat(v=video_input, mode='v')
            video_output = video_output
            A_v_feat.append(video_output)
    A_v_feat = torch.cat(A_v_feat)
    if torch.any(torch.all(A_v_feat == 0, dim=1)):
        raise ValueError("There is at least one row in video_embeds that is all zeros.")
    return A_v_feat



class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x

# the finetuned CAV-MAE model
class CAVMAEFT(nn.Module):
    def __init__(self, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=True):
        super().__init__()
        timm.models.vision_transformer.Block = Block
        print('Use norm_pix_loss: ', norm_pix_loss)

        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(12 - modality_specific_depth)])

        self.norm_a = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

        # print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
        # print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

    def get_patch_num(self, input_shape, stride):
        test_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        test_proj = torch.nn.Conv2d(1, 4, kernel_size=(16, 16), stride=(stride, stride))
        test_output = test_proj(test_input)
        print(test_output.shape)
        return test_output.shape[2], test_output[3], test_output[2] * test_output[2]

    def initialize_weights(self):
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_feat(self, a=None, v=None, mode='av'):
        if mode == 'a':
            # ---- Audio branch ----
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a
            for blk in self.blocks_a:
                a = blk(a)
            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)
            a = a.mean(dim=1)                # mean pool all patches
            a = F.normalize(a, dim=-1)       # l2 normalize
            return a

        elif mode == 'v':
            # ---- Visual branch (多帧处理) ----
            B, N, C, H, W = v.shape
            v = v.view(B * N, C, H, W)
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v
            for blk in self.blocks_v:
                v = blk(v)
            for blk in self.blocks_u:
                v = blk(v, 'v')
            v = self.norm_v(v)
            v = v.view(B, N, v.shape[1], v.shape[2])  # [B, N, num_patches, C]
            v = v.mean(dim=2)                         # patch mean
            v = v.mean(dim=1)                         # frame mean
            v = F.normalize(v, dim=-1)                # l2 normalize
            return v

        elif mode == 'av':
            # ---- Audio branch ----
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a
            for blk in self.blocks_a:
                a = blk(a)
            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)
            a = a.mean(dim=1)
            a = F.normalize(a, dim=-1)

            # ---- Visual branch ----
            B, N, C, H, W = v.shape
            v = v.view(B * N, C, H, W)
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v
            for blk in self.blocks_v:
                v = blk(v)
            for blk in self.blocks_u:
                v = blk(v, 'v')
            v = self.norm_v(v)
            v = v.view(B, N, v.shape[1], v.shape[2])  # [B, N, num_patches, C]
            v = v.mean(dim=2)                         # patch mean
            v = v.mean(dim=1)                         # frame mean
            v = F.normalize(v, dim=-1)

            return a, v
        
    def set_gallery_features(self, gallery_features=None, center = 64):
        self.gallery_features = gallery_features
        
        if self.gallery_features is not None:
            feats = self.gallery_features
            if feats.dtype != torch.float32:
                feats = feats.float()  # Faiss 只支持 float32

            feats_np = feats.detach().cpu().numpy()
            d = feats_np.shape[1]
            k = center

            # GPU KMeans
            kmeans = faiss.Kmeans(d, k, niter=10, verbose=False, gpu=False)
            kmeans.train(feats_np)

            centers = torch.tensor(kmeans.centroids, dtype=feats.dtype, device=feats.device)
            centers = F.normalize(centers, dim=1)

            self.cluster_centers = centers
        
    
    def forward_rest_queue(self, query, device, queue_list, max_queue_size, update_signal, step, args):
        if args.retrieval=='v2a':
            gallery_feat_all=self.gallery_features
            query_feat=self.forward_feat(v=query, mode='v')
        else:
            gallery_feat_all=self.gallery_features
            query_feat=self.forward_feat(a=query, mode='a')

        sim_matrix = query_feat @ gallery_feat_all.t()
        nearest_neighbors_indices = (sim_matrix).argmax(dim=1)
        gallery_feat = gallery_feat_all[nearest_neighbors_indices]

        if (step==0 and update_signal):
            queue_list=update_queue(query_feat, gallery_feat, queue_list, args.con_ratio, max_queue_size, args)

        margin, entropy_queue=get_current_value(queue_list)
        outputs = (query_feat @ gallery_feat.t())
        
        sim_inter = outputs /args.temperature

        loss_L_REM=robust_entropy_minimization_loss(sim_inter, entropy_queue)
        loss_L_U=query_uniformity_loss(query_feat)

        target_query_gallery_gap=compute_query_gallery_gap(query_feat, gallery_feat)
        loss_L_G=(target_query_gallery_gap-margin)**2

        return loss_L_REM, loss_L_U, loss_L_G, queue_list, sim_matrix
    
    def forward_output(self, query, device, args):
        if args.retrieval=='v2a':
            gallery_feat_all=self.gallery_features
            query_feat=self.forward_feat(v=query, mode='v')
        else:
            gallery_feat_all=self.gallery_features
            query_feat=self.forward_feat(a=query, mode='a')

        sim_matrix = query_feat @ gallery_feat_all.t()
        
        sim_inter = sim_matrix/args.temperature
        return sim_inter

    def forward_tta(self, query, device, queue_list, max_queue_size, update_signal, step, args):
        if args.retrieval=='v2a':
            gallery_feat_all=self.gallery_features
            query_feat=self.forward_feat(v=query, mode='v')
        else:
            gallery_feat_all=self.gallery_features
            query_feat=self.forward_feat(a=query, mode='a')

        sim_matrix = query_feat @ gallery_feat_all.t()
        nearest_neighbors_indices = (sim_matrix).argmax(dim=1)
        gallery_feat = gallery_feat_all[nearest_neighbors_indices]
        
        outputs = get_REM_output(query_feat, gallery_feat_all, self.cluster_centers,k=args.candidate_num)

        sim_inter = outputs / args.temperature 
        
        if (step==0 and update_signal):
            queue_list=update_queue_new(query_feat, gallery_feat, sim_inter, queue_list, max_queue_size, args)
            
        margin, entropy_queue =get_current_value(queue_list)
        loss_L_REM, weight=robust_entropy_minimization_loss_with_weight(sim_inter, entropy_queue)
        
        loss_L_RHM = robust_hard_mining_loss(sim_matrix, weight)

        loss_L_U = query_uniformity_loss(query_feat)
        
        target_query_gallery_gap = compute_query_gallery_gap(query_feat, gallery_feat)
        loss_L_G = (target_query_gallery_gap-margin)**2
        
        return loss_L_REM, loss_L_RHM, loss_L_U, loss_L_G, queue_list, sim_matrix
    
    def forward_tta_mixed(self, query, device, queue_list, max_queue_size, update_signal, step, args):
        
        if args.retrieval=='v2a':
            gallery_feat_all=self.gallery_features
            query_feat=self.forward_feat(v=query, mode='v')
        else:
            gallery_feat_all=self.gallery_features
            query_feat=self.forward_feat(a=query, mode='a')

        sim_matrix = query_feat @ gallery_feat_all.t()
        nearest_neighbors_indices = (sim_matrix).argmax(dim=1)
        gallery_feat = gallery_feat_all[nearest_neighbors_indices]
        
        outputs, final_emb = get_REM_output_mixed(query_feat, gallery_feat_all, self.cluster_centers,k=args.candidate_num)

        sim_inter = outputs / args.temperature 
        
        if (step==0 and update_signal):
            queue_list=update_queue_new(query_feat, gallery_feat, sim_inter, queue_list, max_queue_size, args)
            
        margin, entropy_queue =get_current_value(queue_list)
        loss_L_REM, weight=robust_entropy_minimization_loss_with_weight(sim_inter, entropy_queue)
        
        loss_L_RHM = robust_hard_mining_loss(sim_matrix, weight)

        loss_L_U = query_uniformity_loss(query_feat)
        
        target_query_gallery_gap = compute_query_gallery_gap(query_feat, gallery_feat)
        loss_L_G = (target_query_gallery_gap-margin)**2
        
        return loss_L_REM, loss_L_RHM, loss_L_U, loss_L_G, queue_list, sim_matrix, outputs, final_emb
    
    def forward_query_mixed(self, query, device, args):
        if args.retrieval=='v2a':
            query_feat=self.forward_feat(v=query, mode='v')
        else:
            query_feat=self.forward_feat(a=query, mode='a')
        return query_feat

    
def compute_sim_matrix(model, val_loader, device):
    A_a_feat = []
    A_v_feat = []
    model.eval()  # Set the model to evaluation mode

    # Use the `torch.no_grad()` context manager to disable gradient calculation
    with torch.no_grad():
        for a_input, v_input in tqdm(val_loader, desc="Extracting audio and video features"):
            a_input = a_input.to(device)  # Move input data to the correct device
            v_input = v_input.to(device)

            # Use the forward_feat method to get both audio and video embeddings in one pass
            with torch.cuda.amp.autocast():
                audio_output, video_output = model.forward_feat(a=a_input, v=v_input, mode='av')

            # Collect the audio and video features
            A_a_feat.append(audio_output)
            A_v_feat.append(video_output)

    # Concatenate the feature lists to form the complete embeddings
    A_a_feat = torch.cat(A_a_feat, dim=0)  # Concatenate along the batch dimension
    A_v_feat = torch.cat(A_v_feat, dim=0)

    # Compute similarity matrices using matrix multiplication (no need for normalization)
    score_v2a = A_v_feat @ A_a_feat.t()  # Video-to-audio similarity (v2a)
    score_a2v = A_a_feat @ A_v_feat.t()  # Audio-to-video similarity (a2v)
    
    # Convert similarity matrices to numpy arrays for further use
    score_v2a = score_v2a.cpu().numpy()
    score_a2v = score_a2v.cpu().numpy()

    return score_v2a, score_a2v
