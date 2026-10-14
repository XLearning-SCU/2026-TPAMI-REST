import copy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from torch.cuda.amp import autocast
from models.tta_baselines.param import load_model_and_optimizer, copy_model_and_optimizer


class PseudoLabel(nn.Module):
    """
    "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation"
    """
    def __init__(self, algorithm, optimizer, steps, threshold, clf_coeff):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        theta (float) : clf coefficient
        gamma (int) : number of updates
        """
        super().__init__()
        # self.model, self.optimizer = self.configure_model_optimizer(algorithm, alpha=alpha, lr=lr, wd=wd)
        self.model, self.optimizer = algorithm,  optimizer
        self.beta = threshold
        self.theta = clf_coeff
        self.steps = steps
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, device, args, metric_logger, if_adapt=True, counter=None, if_vis=False):
        with autocast():
            if if_adapt:
                if self.episodic:
                    self.reset()

                for _ in range(self.steps):
                    outputs = self.forward_and_adapt(x, device, args, metric_logger, self.model, self.optimizer)
            else:
                # if self.hparams['cached_loader']:
                #     outputs = self.model.classifier(x)
                # else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, device, args, metric_logger, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model.module.forward_output(x, device, args)
        py, y_prime = F.softmax(outputs, dim=-1).max(1)
        flag = py > self.beta
        loss = F.cross_entropy(outputs[flag], y_prime[flag])
        loss.backward()
        optimizer.step()
        metric_logger.update(loss_total=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
           
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)