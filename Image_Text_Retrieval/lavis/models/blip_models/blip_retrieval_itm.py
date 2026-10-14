"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from copy import deepcopy

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
# from lavis.models.albef_models import compute_sim_matrix
from lavis.models.base_model import (
    MomentumDistilationMixin,
    SharedQueueMixin,
    all_gather_with_grad,
    concat_all_gather,
)
from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipSimilarity,
    BlipIntermediateOutput,
)
from lavis.models.med import XBertEncoder
from lavis.models.vit import VisionTransformerEncoder
from torch import nn
import torch.distributed as dist
import utils
from torch.cuda.amp import autocast
import time
import datetime
from losses import *
from models.set_model import *
import lavis.common.dist_utils as dist_utils
import faiss

@registry.register_model("blip_retrieval_itm")
class BlipRetrieval_ITM(BlipBase, MomentumDistilationMixin, SharedQueueMixin):
    """
    BLIP retrieval model.

    Supported model types:
        - coco: fine-tuned BLIP base model on COCO dataset (Karpathy split).
        - flickr: fine-tuned BLIP base model on Flickr30k dataset.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_retrieval", "coco")
        >>> model = load_model("blip_retrieval", "flickr")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "coco": "configs/models/blip_retrieval_coco.yaml",
        "flickr": "configs/models/blip_retrieval_flickr.yaml",
    }

    def __init__(
        self,
        image_encoder,
        text_encoder,
        queue_size,
        alpha=0.4,
        embed_dim=256,
        momentum=0.995,
        negative_all_rank=False,
        max_txt_len=35,
    ):
        """ """
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = image_encoder

        self.text_encoder = text_encoder

        # creating projection layers for ITC
        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)
        
        self.image_features=None
        self.text_features=None
        self.input_ids=None
        self.attention_mask=None
        
    def set_image_features(self, image_features=None, center=64):
        self.image_features = image_features
        
        if self.image_features is not None:
            feats = self.image_features
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
        

    def set_text_features(self,text_features=None,input_ids=None,attention_mask=None,center=64):
        self.text_features = text_features
        self.input_ids=input_ids
        self.input_ids[:,0] = self.tokenizer.enc_token_id
        self.attention_mask=attention_mask
        
        if self.text_features is not None:
            feats = self.text_features
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

    def encode_image(self, image):
        image_embeds = self.visual_encoder(image)      
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]), dim=1)  
        assert self.text_features is not None, "Error: text is None"
        return image_feat, image_embeds

    def encode_text(self, text, device):
        text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(device) 
        
        text_output = self.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]), dim=1)
        assert self.image_features is not None, "Error: image is None"
        return text_feat
    
    def forward_output(self, query, device, args, modality_image_embeds=None):
        if args.retrieval=='i2t':
            gallery_feat_all=self.text_features
            query_feat, query_embeds=self.encode_image(query)
            query_feat=all_gather_with_grad(query_feat)
        else:
            gallery_feat_all=self.image_features
            query_feat=self.encode_text(query, device)
            query_feat=all_gather_with_grad(query_feat)

        sim_matrix = query_feat @ gallery_feat_all.t()
        
        if args.retrieval=='i2t':
            query_embeds=all_gather_with_grad(query_embeds)
            for i in range(sim_matrix.size(0)):
                with torch.no_grad():
                    sims = sim_matrix[i]
                    topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
                    encoder_output = query_embeds[i].repeat(args.k_test,1,1)
                    encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
                    output = self.text_encoder(self.input_ids[topk_idx], 
                            attention_mask = self.attention_mask[topk_idx],
                            encoder_hidden_states = encoder_output,
                            encoder_attention_mask = encoder_att,                             
                            return_dict = True,
                            )
                    score = self.itm_head(output.last_hidden_state[:,0,:])[:,1]
                sim_matrix[i,topk_idx] = score + topk_sim
        else:
            text_input_all = self.tokenizer(query, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
            
            for i in range(sim_matrix.size(0)):
                with torch.no_grad():
                    sims = sim_matrix[i]
                    topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
                    
                    # Fetch required text and image embeddings
                    input_ids_all = text_input_all.input_ids[i].repeat(args.k_test, 1)
                    text_attention_mask = text_input_all.attention_mask[i].repeat(args.k_test, 1)
                    image_embeds = modality_image_embeds[topk_idx.cpu()].to(device)

                    # Create image attention mask for re-ranking
                    image_atts_rerank = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
                    
                    # Pass inputs to the text encoder
                    output_all = self.text_encoder(
                        input_ids_all,
                        attention_mask=text_attention_mask,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts_rerank,
                        return_dict=True,
                    )

                    # Calculate similarity scores
                    score = self.itm_head(output_all.last_hidden_state[:, 0, :])[:, 1]
                sim_matrix[i, topk_idx] = score + topk_sim  # Re-rank the similarity matrix
        
        return sim_matrix / args.temperature
    
    def forward_tta(self, query, device, queue_list, max_queue_size, update_signal, step, args, modality_image_embeds=None):
        if args.retrieval=='i2t':
            gallery_feat_all=self.text_features
            query_feat, query_embeds=self.encode_image(query)
            query_feat=all_gather_with_grad(query_feat)
        else:
            gallery_feat_all=self.image_features
            query_feat=self.encode_text(query, device)
            query_feat=all_gather_with_grad(query_feat)

        sim_matrix = query_feat @ gallery_feat_all.t()
        nearest_neighbors_indices = (sim_matrix).argmax(dim=1)
        gallery_feat = gallery_feat_all[nearest_neighbors_indices]
        
        outputs = get_REM_output(query_feat, gallery_feat_all, self.cluster_centers, k=args.candidate_num)

        sim_inter = outputs / args.temperature 
        
        if (step==0 and update_signal):
            queue_list=update_queue_new(query_feat, gallery_feat, sim_inter, queue_list, max_queue_size, args)
            
        margin, entropy_queue =get_current_value(queue_list)
        loss_L_REM, weight=robust_entropy_minimization_loss_with_weight(sim_inter, entropy_queue)
        
        loss_L_U = query_uniformity_loss(query_feat)

        target_query_gallery_gap = compute_query_gallery_gap(query_feat, gallery_feat)
        loss_L_G = (target_query_gallery_gap-margin)**2
        
        if args.retrieval=='i2t':
            query_embeds=all_gather_with_grad(query_embeds)
            topk_values, topk_indices = sim_matrix.sort(dim=1, descending=True)
            topk_prob = F.softmax(topk_values, dim=1)  # [B, 100] 使用 softmax 获得概率
            idx_pos = topk_indices[:, 0]
            neg_idx_in_topk = torch.multinomial(topk_prob[:, 1:], 1).squeeze(1)
            idx_neg = topk_indices[torch.arange(topk_indices.size(0)), neg_idx_in_topk + 1]
        
            idx_all = torch.cat([idx_pos, idx_neg], dim=0)
            
            input_ids_all = self.input_ids[idx_all]
            attention_mask_all = self.attention_mask[idx_all]
            
            query_embeds_all = query_embeds.repeat(2, 1, 1)
            image_atts = torch.ones(query_embeds_all.size()[:-1], dtype=torch.long).to(device)
            
            output_all = self.text_encoder(
                input_ids_all,
                attention_mask=attention_mask_all,
                encoder_hidden_states=query_embeds_all,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            vl_embeddings = output_all.last_hidden_state[:, 0, :]
            
            vl_output = self.itm_head(vl_embeddings)
            
            target = torch.cat([
                torch.ones(idx_pos.size(0), dtype=torch.long, device=vl_output.device),
                torch.zeros(idx_neg.size(0), dtype=torch.long, device=vl_output.device)
            ], dim=0)
            
            sample_weight = torch.cat([weight, weight], dim=0)
            
            loss_HM = F.cross_entropy(vl_output, target, reduction='none')
            loss_L_RHM = (loss_HM * sample_weight).mean()
            
            if step == args.tta_steps-1:
                with torch.no_grad():
                    if args.retrieval == "i2t":
                        for i in range(sim_matrix.size(0)):
                            sims = sim_matrix[i]
                            topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
                            encoder_output = query_embeds[i].repeat(args.k_test,1,1)
                            encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
                            output = self.text_encoder(self.input_ids[topk_idx], 
                                    attention_mask = self.attention_mask[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
                            score = self.itm_head(output.last_hidden_state[:,0,:])[:,1]
                            sim_matrix[i,topk_idx] = score + topk_sim
        else:
            topk_values, topk_indices = sim_matrix.sort(dim=1, descending=True)
            topk_prob = F.softmax(topk_values, dim=1)
            idx_pos = topk_indices[:, 0]  # Index of positive samples
            neg_idx_in_topk = torch.multinomial(topk_prob[:, 1:], 1).squeeze(1)  # Negative samples from top-k probabilities
            idx_neg = topk_indices[torch.arange(topk_indices.size(0)), neg_idx_in_topk + 1]  # Correct negative indices

            # Combine positive and negative indices
            idx_all = torch.cat([idx_pos, idx_neg], dim=0)

            # Gather the corresponding image embeddings based on idx_all
            modality_image_embeds_all = modality_image_embeds[idx_all.cpu()].to(device)

            # Prepare text inputs for ITM (Image-to-Text Matching)
            text_input_all = self.tokenizer(query, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
            input_ids_all = text_input_all.input_ids.repeat(2, 1)
            attention_mask_all = text_input_all.attention_mask.repeat(2, 1)

            # Create attention mask for image inputs
            image_atts = torch.ones(modality_image_embeds_all.size()[:-1], dtype=torch.long).to(device)

            
            # Pass inputs to the text encoder
            output_all = self.text_encoder(
                input_ids_all,
                attention_mask=attention_mask_all,
                encoder_hidden_states=modality_image_embeds_all,  # Image embeddings as cross-attention input
                encoder_attention_mask=image_atts,  # Attention mask for text embeddings
                return_dict=True,
            )

            # Get the embeddings corresponding to the [CLS] token
            vl_embeddings = output_all.last_hidden_state[:, 0, :]
            vl_output = self.itm_head(vl_embeddings)

            # Target labels: 1 for positive samples, 0 for negative samples
            target = torch.cat([torch.ones(idx_pos.size(0), dtype=torch.long, device=vl_output.device),
                                torch.zeros(idx_neg.size(0), dtype=torch.long, device=vl_output.device)], dim=0)

            # Calculate the sample weights
            sample_weight = torch.cat([weight, weight], dim=0)

            # Compute loss using CrossEntropy
            loss_HM = F.cross_entropy(vl_output, target, reduction='none')

            # Weighted mean of the loss
            loss_L_RHM = (loss_HM * sample_weight).mean()

            # Re-ranking logic for T2I in the last step
            if step == args.tta_steps - 1:
                with torch.no_grad():
                    for i in range(sim_matrix.size(0)):
                        sims = sim_matrix[i]
                        topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
                        
                        # Fetch required text and image embeddings
                        input_ids_all = text_input_all.input_ids[i].repeat(args.k_test, 1)
                        text_attention_mask = text_input_all.attention_mask[i].repeat(args.k_test, 1)
                        image_embeds = modality_image_embeds[topk_idx.cpu()].to(device)

                        # Create image attention mask for re-ranking
                        image_atts_rerank = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
                        
                        # Pass inputs to the text encoder
                        output_all = self.text_encoder(
                            input_ids_all,
                            attention_mask=text_attention_mask,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts_rerank,
                            return_dict=True,
                        )

                        # Calculate similarity scores
                        score = self.itm_head(output_all.last_hidden_state[:, 0, :])[:, 1]
                        sim_matrix[i, topk_idx] = score + topk_sim  # Re-rank the similarity matrix

        return loss_L_REM, loss_L_RHM, loss_L_U, loss_L_G, queue_list, sim_matrix

    def forward_rest_queue(self, query, device, queue_list, max_queue_size, update_signal, step, args, modality_image_embeds=None):
        if args.retrieval=='i2t':
            gallery_feat_all=self.text_features
            query_feat, query_embeds=self.encode_image(query)
            query_feat=all_gather_with_grad(query_feat)
        else:
            gallery_feat_all=self.image_features
            query_feat=self.encode_text(query, device)
            query_feat=all_gather_with_grad(query_feat)

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
        
        if args.retrieval=='i2t':
            query_embeds=all_gather_with_grad(query_embeds)
            if step == args.tta_steps-1:
                with torch.no_grad():
                    if args.retrieval == "i2t":
                        for i in range(sim_matrix.size(0)):
                            sims = sim_matrix[i]
                            topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
                            encoder_output = query_embeds[i].repeat(args.k_test,1,1)
                            encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
                            output = self.text_encoder(self.input_ids[topk_idx], 
                                    attention_mask = self.attention_mask[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
                            score = self.itm_head(output.last_hidden_state[:,0,:])[:,1]
                            sim_matrix[i,topk_idx] = score + topk_sim
        else:

            # Prepare text inputs for ITM (Image-to-Text Matching)
            text_input_all = self.tokenizer(query, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)

            if step == args.tta_steps - 1:
                with torch.no_grad():
                    for i in range(sim_matrix.size(0)):
                        sims = sim_matrix[i]
                        topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
                        
                        # Fetch required text and image embeddings
                        input_ids_all = text_input_all.input_ids[i].repeat(args.k_test, 1)
                        text_attention_mask = text_input_all.attention_mask[i].repeat(args.k_test, 1)
                        image_embeds = modality_image_embeds[topk_idx.cpu()].to(device)

                        # Create image attention mask for re-ranking
                        image_atts_rerank = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
                        
                        # Pass inputs to the text encoder
                        output_all = self.text_encoder(
                            input_ids_all,
                            attention_mask=text_attention_mask,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts_rerank,
                            return_dict=True,
                        )

                        # Calculate similarity scores
                        score = self.itm_head(output_all.last_hidden_state[:, 0, :])[:, 1]
                        sim_matrix[i, topk_idx] = score + topk_sim  # Re-rank the similarity matrix

        return loss_L_REM, loss_L_U, loss_L_G, queue_list, sim_matrix

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    def forward(self, samples):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). The input images.
                - text_input (list): A list of length batch_size, each element is a string of text/caption.
                - image_id (torch.Tensor): A tensor of shape (batch_size, ). The image ids, used to identify same images in batch.
                - epoch (int): The current epoch.
                - iters (int): The current iteration.
                - num_iters_per_epoch (int): The number of iterations per epoch.

        Returns:
            BlipOutput: A BlipOutput object. See ``lavis.models.blip_models.blip_outputs.BlipOutput`` for more details.

        Examples:
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("blip_retrieval", "coco")
            >>> images = torch.randn(4, 3, 384, 384)
            >>> text_input = ["caption of image 1", "another caption of image 1", "caption of image 2", "caption of image 3"]
            >>> image_id = torch.tensor([1, 1, 2, 3])
            >>> samples = {"image": images, "text_input": text_input, "image_id": image_id, "epoch": 0, "iters": 0, "num_iters_per_epoch": 100}
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['sims', 'intermediate_output', 'loss', 'loss_itc', 'loss_itm'])
        """
        image = samples["image"]
        caption = samples["text_input"]
        idx = samples["image_id"]

        alpha = self.alpha * self._rampup_factor(
            epoch=samples["epoch"],
            iters=samples["iters"],
            num_iters_per_epoch=samples["num_iters_per_epoch"],
        )

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder.forward_features(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        text_output = self.text_encoder.forward_text(text)
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # Image-text Contrastive Learning
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(
                self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1
            )
            image_feat_m_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )

            text_output_m = self.text_encoder_m.forward_text(text)
            text_embeds_m = text_output_m.last_hidden_state
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            text_feat_m_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp

            sim_i2t_targets = (
                alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            )
            sim_t2i_targets = (
                alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            )

        sim_i2t = image_feat @ text_feat_m_all / self.temp
        sim_t2i = text_feat @ image_feat_m_all / self.temp

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1
        ).mean()

        loss_itc = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        # Image-text Matching
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # forward the positve image-text pair
        bs = image.size(0)
        output_pos = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        idxs = concat_all_gather(idx)
        if self.negative_all_rank:
            # compute sample similarity
            with torch.no_grad():
                mask = torch.eq(idx, idxs.t())

                image_feat_world = concat_all_gather(image_feat)
                text_feat_world = concat_all_gather(text_feat)

                sim_i2t = image_feat @ text_feat_world.t() / self.temp
                sim_t2i = text_feat @ image_feat_world.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

            image_embeds_world = all_gather_with_grad(image_embeds)

            # select a negative image (from all ranks) for each text
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text (from all ranks) for each image
            input_ids_world = concat_all_gather(encoder_input_ids)
            att_mask_world = concat_all_gather(text.attention_mask)

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])

        else:
            with torch.no_grad():
                mask = torch.eq(idx, idx.t())

                sim_i2t = image_feat @ text_feat.t() / self.temp
                sim_t2i = text_feat @ image_feat.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

            # select a negative image (from same rank) for each text
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text (from same rank) for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder(
            text_ids_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, 0, :],
                output_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )
        itm_logits = self.itm_head(vl_embeddings)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(self.device)
        loss_itm = F.cross_entropy(itm_logits, itm_labels)

        return BlipOutput(
            loss=loss_itc + loss_itm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            sims=BlipSimilarity(
                sim_i2t=sim_i2t,
                sim_t2i=sim_t2i,
                sim_i2t_m=sim_i2t_m,
                sim_t2i_m=sim_t2i_m,
                sim_i2t_targets=sim_i2t_targets,
                sim_t2i_targets=sim_t2i_targets,
            ),
            intermediate_output=BlipIntermediateOutput(
                image_embeds=image_embeds,
                image_embeds_m=image_embeds_m,
                text_embeds=text_embeds,
                text_embeds_m=text_embeds_m,
                encoder_output=output_pos,
                encoder_output_neg=output_neg,
                itm_logits=itm_logits,
                itm_labels=itm_labels,
            ),
        )

    def reset_queue_ptr(self):
        self.queue_ptr = torch.zeros(1, dtype=torch.long)

    @classmethod
    def from_config(cls, cfg=None):
        # set from_pretrained=True to load weights for 'bert-base-uncased'
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        text_encoder = XBertEncoder.from_config(cfg)

        embed_dim = cfg.get("embed_dim", 256)
        momentum = cfg.get("momentum", 0.995)
        alpha = cfg.get("alpha", 0.4)
        negative_all_rank = cfg.get("negative_all_rank", False)

        queue_size = cfg.get("queue_size", 0)
        max_txt_len = cfg.get("max_txt_len", 35)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            queue_size=queue_size,
            alpha=alpha,
            embed_dim=embed_dim,
            momentum=momentum,
            negative_all_rank=negative_all_rank,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)
        model.reset_queue_ptr()
        
        # model = model.half()
        convert_weights_to_bf16(model)
        # convert_weights_to_int8(model)

        return model

    def compute_sim_matrix(self, test_loader_text, test_loader_image, device, args):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """

        return compute_sim_matrix(self, test_loader_text, test_loader_image, device, args)

@torch.no_grad()
def compute_sim_matrix(model, test_loader_text, test_loader_image, device, args): #(model, data_loader, **kwargs):
    k_test = args.k_test

    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    with autocast():
        texts = test_loader_text.dataset.text

        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        for i in range(0, num_text, text_bs):
            text = texts[i : min(num_text, i + text_bs)]
            text_input = model.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(model.device)
            text_output = model.text_encoder.forward_text(text_input)
            text_embed = F.normalize(
                model.text_proj(text_output.last_hidden_state[:, 0, :])
            )
            text_embeds.append(text_embed)
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

        text_embeds = torch.cat(text_embeds, dim=0)
        text_ids = torch.cat(text_ids, dim=0)
        text_atts = torch.cat(text_atts, dim=0)
        if hasattr(model.tokenizer, "enc_token_id"):
            text_ids[:, 0] = model.tokenizer.enc_token_id

        image_feats = []
        image_embeds = []
        for image, img_id in test_loader_image:
            image = image.to(device)
            
            image_feat = model.visual_encoder.forward_features(image)
            image_embed = model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            image_feats.append(image_feat.cpu())
            image_embeds.append(image_embed)

        image_feats = torch.cat(image_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        sims_matrix = image_embeds @ text_embeds.t()

        # Batched version of i2t and t2i scoring
        batch_size = 20  # or any number that fits your GPU memory

        # ----------- I2T (image -> text) --------------
        score_matrix_i2t = torch.full(
            (len(test_loader_image.dataset.image), len(texts)),
            -100.0,
            dtype=torch.float16,
            device=device,
        )

        num_tasks = utils.get_world_size()
        rank = utils.get_rank()
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        sims_subset = sims_matrix[start:end]  # [num_images_local, num_texts]
        for i, batch_sims in enumerate(metric_logger.log_every(sims_subset.split(batch_size), 100, "I2T Scoring")):
            batch_sims = batch_sims.to(device)
            bsz = batch_sims.size(0)
            topk_sim, topk_idx = batch_sims.topk(k=k_test, dim=1)  # [B, k]

            # Image features
            encoder_output = image_feats[start + i * batch_size : start + i * batch_size + bsz].to(device)  # [B, L, D]
            encoder_output = encoder_output.unsqueeze(1).expand(-1, k_test, -1, -1)
            encoder_output = encoder_output.reshape(-1, encoder_output.size(-2), encoder_output.size(-1))  # [B*k, L, D]
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

            # Text inputs
            selected_text_ids = text_ids[topk_idx]        # [B, k, L]
            selected_text_ids = selected_text_ids.reshape(-1, selected_text_ids.size(-1))
            selected_text_atts = text_atts[topk_idx].reshape(-1, text_atts.size(-1))

            output = model.text_encoder(
                selected_text_ids,
                attention_mask=selected_text_atts,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]  # [B*k]
            score = score.view(bsz, k_test)
            score_matrix_i2t[start + i * batch_size : start + i * batch_size + bsz, topk_idx] = score + topk_sim


        # ----------- T2I (text -> image) --------------
        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full(
            (len(texts), len(test_loader_image.dataset.image)),
            -100.0,
            dtype=score_matrix_i2t.dtype,
            device=device,
        )

        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)
        sims_subset = sims_matrix[start:end]  # [num_texts_local, num_images]

        for i, batch_sims in enumerate(metric_logger.log_every(sims_subset.split(batch_size), 100, "T2I Scoring")):
            batch_sims = batch_sims.to(device)
            bsz = batch_sims.size(0)
            topk_sim, topk_idx = batch_sims.topk(k=k_test, dim=1)  # [B, k]

            encoder_output = image_feats[topk_idx.cpu()].to(device)  # [B, k, L, D]
            encoder_output = encoder_output.reshape(-1, encoder_output.size(-2), encoder_output.size(-1))
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

            text_input_ids = text_ids[start + i * batch_size : start + i * batch_size + bsz].to(device)  # [B, L]
            text_input_atts = text_atts[start + i * batch_size : start + i * batch_size + bsz].to(device)
            text_input_ids = text_input_ids.unsqueeze(1).expand(-1, k_test, -1).reshape(-1, text_input_ids.size(-1))
            text_input_atts = text_input_atts.unsqueeze(1).expand(-1, k_test, -1).reshape(-1, text_input_atts.size(-1))

            output = model.text_encoder(
                text_input_ids,
                attention_mask=text_input_atts,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score = score.view(bsz, k_test)
            score_matrix_t2i[start + i * batch_size : start + i * batch_size + bsz, topk_idx] = score + topk_sim


    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

@torch.no_grad()
def compute_sim_matrix_ddp(model, test_loader_text, test_loader_image, device, args): #(model, data_loader, **kwargs):
    k_test = args.k_test

    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    with autocast():
        texts = test_loader_text.dataset.text

        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        for i in range(0, num_text, text_bs):
            text = texts[i : min(num_text, i + text_bs)]
            text_input = model.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(model.device)
            text_output = model.text_encoder.forward_text(text_input)
            text_embed = F.normalize(
                model.text_proj(text_output.last_hidden_state[:, 0, :])
            )
            text_embeds.append(text_embed)
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

        text_embeds = torch.cat(text_embeds, dim=0)
        text_ids = torch.cat(text_ids, dim=0)
        text_atts = torch.cat(text_atts, dim=0)
        if hasattr(model.tokenizer, "enc_token_id"):
            text_ids[:, 0] = model.tokenizer.enc_token_id

        image_feats = []
        image_embeds = []
        for image, img_id in test_loader_image:
            image = image.to(device)
            
            image_feat = model.visual_encoder.forward_features(image)
            image_embed = model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            image_feats.append(image_feat.cpu())
            image_embeds.append(image_embed)

        image_feats = torch.cat(image_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        sims_matrix = image_embeds @ text_embeds.t()
        
        # v2
        score_matrix_i2t = torch.full((len(test_loader_image.dataset.image), len(texts)), -100.0).to(device)
        score_matrix_i2t = torch.full(
            (len(test_loader_image.dataset.image), len(texts)),
            -100.0,
            dtype=torch.float16,  # ✅ 指定 float16 精度
            device=device
        )

        num_tasks = utils.get_world_size()
        rank = utils.get_rank() 
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
        ):
            # topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

            encoder_output = image_feats[start + i].repeat(k_test, 1, 1).to(model.device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                model.device
            )
            output = model.text_encoder(
                text_ids[topk_idx],
                attention_mask=text_atts[topk_idx],
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_i2t[start + i, topk_idx] = score + topk_sim

        sims_matrix = sims_matrix.t()
        # score_matrix_t2i = torch.full((len(texts), len(test_loader_image.dataset.image)), -100.0).to(device)
        score_matrix_t2i = torch.full(
            (len(texts), len(test_loader_image.dataset.image)),
            -100.0,
            dtype=score.dtype,   # ✅ 确保一致精度
            device=device
        )

        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 100, header)
        ):

            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
            encoder_output = image_feats[topk_idx.cpu()].to(model.device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                model.device
            )
            output = model.text_encoder(
                text_ids[start + i].repeat(k_test, 1),
                attention_mask=text_atts[start + i].repeat(k_test, 1),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
