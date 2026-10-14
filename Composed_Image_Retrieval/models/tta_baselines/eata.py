"""
Copyright to EATA ICML 2022 Authors, 2022.03.20
Based on Tent ICLR 2021 Spotlight. 
"""
import math

import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from models.tta_baselines.param import load_model_and_optimizer, copy_model_and_optimizer

class EATA(nn.Module):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, fishers=None, fisher_alpha=2000.0, episodic=False, e_margin=math.log(1000)/2-1, d_margin=0.05):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = e_margin # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = d_margin # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

        self.fishers = fishers # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        self.fisher_alpha = fisher_alpha # trade-off \beta for two losses (Eqn. 8) 

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, device, args, metric_logger, if_adapt=True, counter=None, if_vis=False):
        if self.episodic:
            self.reset()

        with autocast():
            #if self.steps > 0:
            if if_adapt:
                for _ in range(self.steps):
                    outputs, num_counts_2, num_counts_1, updated_probs = forward_and_adapt_eata(x, device, args, metric_logger, self.model, self.optimizer, self.fishers, self.e_margin, self.current_model_probs, fisher_alpha=self.fisher_alpha, num_samples_update=self.num_samples_update_2, d_margin=self.d_margin)
                    self.num_samples_update_2 += num_counts_2
                    self.num_samples_update_1 += num_counts_1
                    self.reset_model_probs(updated_probs)
            else:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(x)
        
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_eata(x, device, args, metric_logger, model, optimizer, fishers, e_margin, current_model_probs, fisher_alpha=50.0, d_margin=0.05, scale_factor=2, num_samples_update=0):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    Return: 
    1. model outputs; 
    2. the number of reliable and non-redundant samples; 
    3. the number of reliable samples;
    4. the moving average  probability vector over all previous samples
    """
    # forward
    outputs = model.forward_output(x, device, args)
    # adapt
    entropys = softmax_entropy(outputs)
    
    ori_entropys=entropys
    # filter unreliable samples
    # filter_ids_1 = torch.where(entropys < e_margin)

    k = int(len(entropys) * e_margin)
    top_k_values, _ = torch.topk(entropys, k, largest=False)
    threshold = top_k_values[-1]  # The k-th smallest entropy value
    filter_ids_1 = torch.where(entropys <= threshold)

    ids1 = filter_ids_1
    ids2 = torch.where(ids1[0]>-0.1)
    entropys = entropys[filter_ids_1] 
    # filter redundant samples
    if current_model_probs is not None: 
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        entropys = entropys[filter_ids_2]
        ids2 = filter_ids_2
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
    else:
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin)).clamp(min=1e-6, max=1e6)
    # implementation version 1, compute loss, all samples backward (some unselected are masked)
    entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
    if entropys.numel() == 0 or torch.isnan(entropys).any() or torch.isinf(entropys).any():
        loss=ori_entropys.mean(0)
    else:
        loss = entropys.mean(0)

    """
    # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
    # if x[ids1][ids2].size(0) != 0:
    #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
    """
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
        loss += ewc_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    metric_logger.update(loss_total=loss.item())
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)