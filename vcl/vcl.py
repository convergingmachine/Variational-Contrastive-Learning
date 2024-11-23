import math
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torchvision.models import resnet50

from lightly.loss.ntx_ent_loss import NTXentLoss
from vcl.beta_ntx_ent_loss import betaNTXentLoss
from vcl.proj_head import VCLProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler


class VCL(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int, temperature: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = VCLProjectionHead()
        self.criterion = betaNTXentLoss(beta=0.005, temperature=0.07) #NTXentLoss(temperature=temperature, gather_distributed=True)
        
        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

    def compute_m_params(self, mu_i, mu_j, sigma2_i, sigma2_j):
        # Compute m_mu as the mean of i_mu and j_mu
        m_mu = (mu_i + mu_j) / 2
        
        # Compute m_sigma as the square root of the product of sigma_i and sigma_j
        m_sigma = torch.sqrt(sigma2_i * sigma2_j)

        return m_mu, m_sigma

    def similarity_loss(self, mu0: Tensor, mu1: Tensor, logVar0: Tensor, logVar1: Tensor) -> Tensor:
        # Compute the mean and sigma for the two views
        sigma20, sigma21 = logVar0.exp(), logVar1.exp()
        mum, sigma2m = self.compute_m_params(mu0, mu1, sigma20, sigma21)
        logVarm = torch.log(sigma2m)
        
        # Compute the first term: -0.5 * (log(sigma_i) - log(sigma_j))
        log_diffs = (logVarm - logVar0) + (logVarm - logVar1) 
        term1 = 0.5 * log_diffs
        
        # Compute the second term: 0.25 * ((mu_i - mu_m)^2 + (mu_j - mu_m)^2) / sigma_m^2
        diff_i = mu0 - mum
        diff_j = mu1 - mum
        term2 = 0.25 * ((diff_i**2 + diff_j**2) / (sigma2m))
        
        # Combine both terms
        l_dist_value = term1 + term2
        
        return l_dist_value.mean()
    
    def normalizing_loss(self, mu: Tensor, logVar: Tensor) -> Tensor:
        term3  = - 0.5 * (1 + logVar - torch.exp(logVar) - mu**2)
        return term3.mean()

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        views, targets = batch[0], batch[1]
        features = self.forward(torch.cat(views)).flatten(start_dim=1)
        z, mu, logVar = self.projection_head(features)
        
        z0, z1 = z.chunk(len(views))
        mu0, mu1 = mu.chunk(len(views))
        logVar0, logVar1 = logVar.chunk(len(views))

        loss = self.criterion(z0, z1) + self.similarity_loss(mu0, mu1, logVar0, logVar1) + self.normalizing_loss(mu, logVar)
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        cls_loss, cls_log = self.online_classifier.training_step(
            (features.detach(), targets.repeat(len(views))), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        optimizer = LARS(
            [
                {"name": "vcl", "params": params},
                {
                    "name": "vcl_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Square root learning rate scaling improves performance for small
            # batch sizes (<=2048) and few training epochs (<=200). Alternatively,
            # linear scaling can be used for larger batches and longer training:
            #   lr=0.3 * self.batch_size_per_device * self.trainer.world_size / 256
            # See Appendix B.1. in the SimCLR paper https://arxiv.org/abs/2002.05709
            lr=0.075 * math.sqrt(self.batch_size_per_device * self.trainer.world_size),
            momentum=0.9,
            # Note: Paper uses weight decay of 1e-6 but reference code 1e-4. See:
            # https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/README.md?plain=1#L103
            weight_decay=1e-6,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
