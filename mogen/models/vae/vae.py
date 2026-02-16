import sys
sys.path.append('.')

import torch
import torch.nn as nn
from collections import OrderedDict

from mogen.models.vae.encdec import Encoder, Decoder

class VAE(nn.Module):
    def __init__(self,
                 input_width : int = 263,
                 output_emb_width : int = 32, # latent dim
                 down_t : int = 3,
                 stride_t : int = 2,
                 width : int = 256, # conv 1d width
                 depth : int = 3,
                 dilation_growth_rate : int = 3,
                 activation : str = 'relu',
                 norm = None,
                 pad_mode : str = 'replicate',
                 ae : bool = False,
                 ) -> None:
        super(VAE, self).__init__()

        self.ae = ae

        self.latent_dim = output_emb_width

        post_proj_dim = 2*self.latent_dim if not self.ae else self.latent_dim
        self.post_proj = nn.Linear(width, post_proj_dim)


        self.encoder = Encoder(input_width, down_t, stride_t, width, depth,
                           dilation_growth_rate, pad_mode=pad_mode, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, width, depth,
                               dilation_growth_rate, pad_mode=pad_mode, activation=activation, norm=norm)


    def pre_process(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def post_process(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features):
        if not self.ae:
            z, dist = self.encode(features)
            feats_rst = self.decode(z)
            return feats_rst, z, dist
        else:
            z = self.ae_encode(features)
            feats_rst = self.decode(z)
            return feats_rst, z

    def ae_encode(self, features):
        x = self.pre_process(features)  # [N, input_width, T]

        latent = self.encoder(x)  # [bs, width, T//4]
        latent = self.post_process(latent)  # [bs, T//4, width]
        latent = self.post_proj(latent) # [bs, T//4, latent_dim]
        return latent

    def encode(self, features):
        x = self.pre_process(features) # [N, input_width, T]

        dist = self.encoder(x) # [bs, width, T//4]
        dist = self.post_process(dist)  # [bs, T//4, width]
        dist = self.post_proj(dist) # [bs, T//4, 2*latent_dim]

        mu, log_var = torch.chunk(dist, 2, dim=-1)
        log_var = torch.clamp(log_var, -30.0, 20.0)
        std = log_var.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample() # [bs, T//4, latent_dim]

        return latent, dist

    def decode(self, z):
        z = self.pre_process(z)
        feats = self.decoder(z)
        return feats

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint['state_dict']
        if hasattr(self, 'text_encoder'):
            clip_k = []
            for k, v in state_dict.items():
                if 'text_encoder' in k:
                    clip_k.append(k)
            for k in clip_k:
                del checkpoint['state_dict'][k]

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        if hasattr(self, 'text_encoder'):
            clip_state_dict = self.text_encoder.state_dict()
            new_state_dict = OrderedDict()
            for k, v in clip_state_dict.items():
                new_state_dict['text_encoder.' + k] = v
            for k, v in state_dict.items():
                if 'text_encoder' not in k:
                    new_state_dict[k] = v
            return super().load_state_dict(new_state_dict, strict)
        else:
            return super().load_state_dict(state_dict, strict)