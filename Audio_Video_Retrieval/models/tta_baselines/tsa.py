import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from models.tta_baselines.param import load_model_and_optimizer, copy_model_and_optimizer

class TSA(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, device, args, metric_logger, if_adapt=True, counter=None, if_vis=False):
        if self.episodic:
            self.reset()
            
        with autocast():
            if if_adapt:
                #print("adaptating..")
                for _ in range(self.steps):
                    outputs = forward_and_adapt(x, device, args, metric_logger, self.model, self.optimizer)
            else:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


def FixMatchLoss(logits_s,logits_w,threshold=0.95):
        # Pseudo-labeling for unlabeled data
        with torch.no_grad():
            # Get the maximum probabilities and the pseudo-labels
            probs_u = F.softmax(logits_w, dim=-1)
            max_probs, pseudo_labels = torch.max(probs_u, dim=-1)

            # Create a mask for confident pseudo-labels
            mask = max_probs.ge(threshold).float()

        # Unsupservised loss for unlabeled data
        loss_u = (F.cross_entropy(logits_s, pseudo_labels) * mask).mean()
        return loss_u,mask

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, device, args, metric_logger, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model.forward_output(x, device, args)
    # adapt
    p_sum = outputs.softmax(dim=-1).sum(dim=-2)
    loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()
    loss = FixMatchLoss(outputs, outputs.detach(), threshold=0.85)[0].mean() - loss_bal 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    metric_logger.update(loss_total=loss.item())
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return outputs