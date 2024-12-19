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


class SURGICAL(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.transforms = get_tta_transforms(self.dataset_name)
        self.eps = 1e-8        

        self.entropy_min = 0.4 * math.log(self.num_classes)
        self.lambd = cfg.SURGICAL.LAMBDA
        

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs = x[0]
        device = imgs.device

        logits = self.model(imgs)
        logits_aug = self.model(self.transforms(imgs))

        self.optimizer.zero_grad()
        loss = entropy_minmization(logits, self.entropy_min) + self.lambd * consistency(logits, logits_aug)
        loss.backward()
        self.optimizer.step()

        return logits


    def configure_model(self):
        self.model.train()
        self.model.requires_grad_(False)

        for nm, m in self.model.named_modules():
            if 'layer1' in nm or 'block1' in nm or 'stage_1' in nm: # first block (configurable)
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
def consistency(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    """Consistency loss between two softmax distributions."""
    return -(x.softmax(1) * y.log_softmax(1)).sum(1).mean()

def entropy_minmization(outputs,e_margin = 0.4):
    """Calculate entropy of the output of a batch of images.
    """
    entropys = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
    filter_ids_1 = torch.where(entropys < e_margin)
    entropys = entropys[filter_ids_1]
    ent = entropys.mean(0)
    return ent