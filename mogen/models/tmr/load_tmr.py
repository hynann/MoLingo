import sys
sys.path.append('.')

from mogen.models.tmr.actor import ACTORStyleEncoder, ACTORStyleDecoder
from mogen.models.tmr.tmr import TMR
from mogen.models.tmr.losses import InfoNCE_with_filtering
from mogen.models.tmr.text import TokenEmbeddings
import os
import torch

class Normalizer:
    def __init__(self, base_dir: str, eps: float = 1e-12, disable: bool = False):
        self.base_dir = base_dir
        self.mean_path = os.path.join(base_dir, "mean.pt")
        self.std_path = os.path.join(base_dir, "std.pt")
        self.eps = eps

        self.disable = disable
        if not disable:
            self.load()

    def load(self):
        self.mean = torch.load(self.mean_path)
        self.std = torch.load(self.std_path)

    def save(self, mean, std):
        os.makedirs(self.base_dir, exist_ok=True)
        torch.save(mean, self.mean_path)
        torch.save(std, self.std_path)

    def __call__(self, x):
        if self.disable:
            return x
        x = (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)
        return x

    def inverse_hml3d(self, x):
        x = x * (self.std.to(x.device) + self.eps) + self.mean.to(x.device)
        return x

    def inverse(self, x):
        if self.disable:
            return x
        x = x * (self.std + self.eps) + self.mean
        return x

def load_tmr_model(vae=True, nfeats=263):
    motion_encoder = ACTORStyleEncoder(
        nfeats=nfeats,
        vae=vae,
        latent_dim=256,
        ff_size=1024,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
        activation='gelu'
    )

    text_encoder = ACTORStyleEncoder(
        nfeats=768,
        vae=vae,
        latent_dim=256,
        ff_size=1024,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
        activation='gelu'
    )

    motion_decoder = ACTORStyleDecoder(
        nfeats=nfeats,
        latent_dim=256,
        ff_size=1024,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
        activation='gelu'
    )

    info = InfoNCE_with_filtering(
        temperature=0.1,
        threshold_selfsim=0.8
    )

    tmr_model = TMR(
        motion_encoder=motion_encoder,
        text_encoder=text_encoder,
        motion_decoder=motion_decoder,
        vae=vae,
        contrastive_loss=info,
        # temperaure=0.1,
        # threshold_selfsim=0.8,
    )

    root_dir = './mogen/checkpoints/TMR'

    text_to_token_emb = TokenEmbeddings(
        path = os.path.join(root_dir, 'datasets/annotations/humanml3d'),
        modelname='distilbert-base-uncased',
        preload=True
    )

    normalizer = Normalizer(
        base_dir= os.path.join(root_dir, 'stats/humanml3d/guoh3dfeats'),
        eps=1e-12
    )

    return tmr_model, text_to_token_emb, normalizer