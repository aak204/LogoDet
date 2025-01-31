import torch.nn as nn
import torch.nn.functional as F
import timm

class LogoEncoder(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.backbone = timm.create_model('lcnet_050.ra2_in1k', 
                                          pretrained=True, num_classes=0)
        self.proj = nn.Linear(1280, emb_dim)

    def forward(self, x):
        x = self.backbone(x)     # [B, 1280]
        x = self.proj(x)         # [B, emb_dim]
        x = F.normalize(x, dim=1)
        return x
