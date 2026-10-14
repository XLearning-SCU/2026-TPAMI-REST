import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from models.tta_baselines.param import load_model_and_optimizer, copy_model_and_optimizer


class REST(nn.Module):
    """Internal REST implementation for the OQS setting."""

    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "REST requires >= 1 adaptation step."
        self.episodic = episodic
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, query_reference, query_instruction, args, metric_logger, queue_list, max_queue_size, update_signal):
        if self.episodic:
            self.reset()
        with autocast():
            for step in range(self.steps):
                queue_list, outputs = forward_and_adapt(
                    query_reference,
                    query_instruction,
                    args,
                    metric_logger,
                    queue_list,
                    max_queue_size,
                    update_signal,
                    step,
                    self.model,
                    self.optimizer,
                )
        return queue_list, outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)


@torch.enable_grad()
def forward_and_adapt(query_reference, query_instruction, args, metric_logger, queue_list, max_queue_size, update_signal, step, model, optimizer):
    loss_L_REM, loss_L_RHM, loss_L_U, loss_L_G, queue_list, outputs = model.forward_tta(
        query_reference, query_instruction, queue_list, max_queue_size, update_signal, step, args
    )

    loss_L_C = loss_L_REM + loss_L_RHM
    # Keep the original left-to-right floating-point addition order.
    loss_REST = loss_L_REM + loss_L_RHM + loss_L_U + loss_L_G
    if args.evaluation:
        loss_REST = loss_REST * 0

    optimizer.zero_grad()
    loss_REST.backward()
    optimizer.step()

    metric_logger.update(loss_L_REM=loss_L_REM.item())
    metric_logger.update(loss_L_RHM=loss_L_RHM.item())
    metric_logger.update(loss_L_C=loss_L_C.item())
    metric_logger.update(loss_L_U=loss_L_U.item())
    metric_logger.update(loss_L_G=loss_L_G.item())
    metric_logger.update(loss_REST=loss_REST.item())
    metric_logger.update(loss_total=loss_REST.item())
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return queue_list, outputs
