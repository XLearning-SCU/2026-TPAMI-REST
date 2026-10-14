import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class REST_DQS(nn.Module):
    """Internal REST implementation for the DQS setting."""

    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "REST requires >= 1 adaptation step."
        self.episodic = episodic

        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def forward(self, query, device, args, metric_logger, queue_list, max_queue_size, update_signal):
        with autocast():
            for step in range(self.steps):
                queue_list, outputs = forward_and_adapt(
                    query,
                    device,
                    args,
                    metric_logger,
                    queue_list,
                    max_queue_size,
                    update_signal,
                    step,
                    self.model,
                    self.teacher_model,
                    self.optimizer,
                )
        return queue_list, outputs


@torch.enable_grad()
def forward_and_adapt(query, device, args, metric_logger, queue_list, max_queue_size, update_signal, step, model, teacher_model, optimizer):
    loss_L_REM, loss_L_RHM, loss_L_U, loss_L_G, queue_list, outputs, stu_logits, final_emb = model.forward_tta_mixed(
        query, device, queue_list, max_queue_size, update_signal, step, args
    )

    loss_L_C = loss_L_REM + loss_L_RHM
    # Keep the original left-to-right floating-point addition order.
    loss_REST = loss_L_REM + loss_L_RHM + loss_L_U + loss_L_G

    log_stu_prob = F.log_softmax(stu_logits / args.temperature, dim=-1)
    with torch.no_grad():
        query_emb = teacher_model.forward_query_mixed(query, device, args)
        teacher_logits = torch.bmm(query_emb.unsqueeze(1), final_emb.transpose(1, 2)).squeeze(1)
        teacher_prob = F.softmax(teacher_logits / args.temperature, dim=-1)
        log_teacher_prob = F.log_softmax(teacher_logits / args.temperature, dim=-1)

    max_confidence, _ = teacher_prob.max(dim=1)
    keep_mask = max_confidence >= torch.quantile(max_confidence, 0.7)
    kl_per_sample = (teacher_prob * (log_teacher_prob - log_stu_prob)).sum(dim=1)
    loss_D_KL = kl_per_sample[keep_mask].mean() if keep_mask.any() else kl_per_sample.mean()

    if getattr(args, "gradient_decoupling", False):
        gradient_decoupling_update(model, optimizer, loss_REST, loss_D_KL)
    else:
        standard_update(model, optimizer, loss_REST)

    metric_logger.update(loss_L_REM=loss_L_REM.item())
    metric_logger.update(loss_L_RHM=loss_L_RHM.item())
    metric_logger.update(loss_L_C=loss_L_C.item())
    metric_logger.update(loss_L_U=loss_L_U.item())
    metric_logger.update(loss_L_G=loss_L_G.item())
    metric_logger.update(loss_REST=loss_REST.item())
    metric_logger.update(loss_D_KL=loss_D_KL.item())
    metric_logger.update(loss_total=loss_REST.item())
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return queue_list, outputs


def standard_update(model, optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def gradient_decoupling_update(model, optimizer, loss_REST, loss_D_KL):
    weight = 1.0 / (1.0 + loss_D_KL.item())

    optimizer.zero_grad()
    loss_D_KL.backward(retain_graph=True)
    general_grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

    optimizer.zero_grad()
    loss_REST.backward()

    for p, general_grad in zip(model.parameters(), general_grads):
        domain_grad = p.grad
        if domain_grad is None or general_grad is None:
            continue

        general_hat = general_grad / (torch.norm(general_grad) + 1e-12)
        domain_hat = domain_grad / (torch.norm(domain_grad) + 1e-12)
        parallel = torch.dot(domain_grad.flatten(), general_hat.flatten()) * general_hat
        orthogonal = domain_grad - parallel

        if torch.dot(domain_hat.flatten(), general_hat.flatten()) < 0:
            p.grad = weight * orthogonal
        else:
            p.grad = weight * (orthogonal + parallel)

    optimizer.step()
