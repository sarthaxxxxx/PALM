import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms
from collections import defaultdict


class OURS(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.base_lr = self.optimizer.param_groups[0]['lr']
        self.betas = self.optimizer.param_groups[0]['betas']
        self.weight_decay = self.optimizer.param_groups[0]['weight_decay']
        self.transforms = get_tta_transforms(self.dataset_name)
        self.eps = 1e-8        
        
        self.trainable_dict = {k: v for k, v in self.model.named_parameters() if v.requires_grad}
        self.exp_sens = {np: 0.0 for np in self.trainable_dict.keys()}
        self.grad_weight = {np: 0.0 for np in self.trainable_dict.keys()}

        self.beta_3 = cfg.OURS.BETA3
        self.threshold = cfg.OURS.THRESH
        self.temp = cfg.OURS.TEMP
        self.lambd = cfg.OURS.LAMBDA

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        logsoftmax = nn.LogSoftmax(dim = -1)

        conf = {}

        imgs = x[0]
        device = imgs.device

        logits = self.model(imgs)
        logits_aug = self.model(self.transforms(imgs))

        self.model.zero_grad()
        uniform_dist = torch.ones((imgs.shape[0], self.num_classes)).to(device) # uniform distribution
        logits_copy = logits / self.temp # temperature scaling
        loss = torch.mean(torch.sum(-uniform_dist * logsoftmax(logits_copy), dim=-1)) # KL divergence ~ Cross-entropy
        loss.backward(retain_graph = True)
        
        for n, p in self.trainable_dict.items():
            layer_grad = p.grad.data
            score = torch.norm(layer_grad, p = 1).cpu().numpy() # score for each layer
            if score <= self.threshold: 
                conf[n] = layer_grad # if score is less than threshold, then the layer is considered as uncertain

        for np, param in self.trainable_dict.items():
            if np in conf:
                sens = (param.data * conf[np]).abs() # sensitivity
                self.exp_sens[np] = (self.beta_3 * self.exp_sens[np]) + (1 - self.beta_3) * sens # moving average of sensitivity
                uncertainty = (sens - self.exp_sens[np]).abs() # uncertainty
                self.grad_weight[np] = uncertainty


        params = []
        for k, v in self.grad_weight.items():
            u_value =  v
            step_size = (u_value + self.eps) / (self.exp_sens[k] + self.eps) # learning rate importance
            step_size = step_size.mean().item() if isinstance(step_size, torch.Tensor)\
                        else 0.0
            params.append({"params": self.trainable_dict[k],
                            "lr": self.base_lr*step_size,
                            "betas": self.betas,
                            "weight_decay": self.weight_decay})
    
        self.optimizer = torch.optim.Adam(params)
        self.optimizer.zero_grad()
        loss = entropy_minmization(logits) + self.lambd*logits.shape[1]*consistency(logits, logits_aug)
        loss.backward()
        self.optimizer.step()
        return logits

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.Conv2d)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")           
        return params, names

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, nn.Conv2d):
                m.train()
                m.requires_grad_(True)


    @staticmethod
    def check_model(model):
        """Check model for compatability with law."""
        is_training = model.training
        assert is_training, "law needs train mode: call model.train()"

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()

@torch.jit.script
def consistency(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    """Consistency loss between two softmax distributions."""
    return -(x.softmax(1) * y.log_softmax(1)).sum(1).mean()

def entropy_minmization(outputs, e_margin = 0.4):
    """Calculate entropy of the output of a batch of images.
    """
    entropys = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
    filter_ids_1 = torch.where(entropys < e_margin)
    entropys = entropys[filter_ids_1]
    ent = entropys.mean(0)
    return ent