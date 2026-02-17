import sys
sys.path.append('.')

import math
import torch
import torch.nn as nn
from functools import partial
from transformers import T5EncoderModel, T5Tokenizer

from mogen.models.molingo.flowloss import FLowLoss
from mogen.models.operator.position_encoding import build_position_encoding

class MoLingo(nn.Module):
    def __init__(self,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4.,
                 vae_embed_dim=256,
                 label_drop_prob=0.1,
                 proj_dropout=0.1,
                 flowloss_d=12,
                 flowloss_w=1536,
                 flow_batch_mul=4,
                 unit_length=4,
                 grad_checkpointing=False,
                 token_size=49,
                 sample_steps=32,
                 t5_max_len=64,
                 adapter_layers=6,
                 ae=False,
                 ):
        super().__init__()

        self.ae = ae
        self.seq_len = token_size
        self.token_embed_dim = vae_embed_dim
        self.grad_checkpointing = grad_checkpointing
        self.unit_length = unit_length
        self.label_drop_prob = label_drop_prob
        self.t5_dim = 1024
        self.flow_batch_mul = flow_batch_mul
        self.encode_text = self.t5_encode_text

        self.cond_proj = nn.Linear(self.t5_dim, decoder_embed_dim)
        self.text_aligner = TextAdapter(n_layers=adapter_layers,
                            d_model=decoder_embed_dim,
                            n_head=decoder_num_heads).to('cuda')

        self.t5_tok = T5Tokenizer.from_pretrained("t5-large")
        self.t5_model = T5EncoderModel.from_pretrained("t5-large")
        self.t5_model.to('cuda')
        self.t5_model.eval()
        for p in self.t5_model.parameters(): p.requires_grad_(False)
        self.t5_max_len = t5_max_len


        self.z_proj = nn.Linear(self.token_embed_dim, decoder_embed_dim, bias=True)
        self.query_pos_encoder = build_position_encoding(decoder_embed_dim, batch_first=True)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=decoder_embed_dim,
                                                          nhead=decoder_num_heads,
                                                          dim_feedforward=int(mlp_ratio * decoder_embed_dim),
                                                          dropout=proj_dropout,
                                                          activation="gelu",
                                                          batch_first=True)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=decoder_depth)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, vae_embed_dim))
        self.initialize_weights()

        self.flow_loss = FLowLoss(
            target_channels=vae_embed_dim,
            z_channels=decoder_embed_dim,
            width=flowloss_w,
            depth=flowloss_d,
            grad_checkpointing=grad_checkpointing,
            sample_steps=sample_steps,
        )


    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)


    def t5_encode_text(self, raw_text):
        res_batch = self.t5_tok(
            text=raw_text,
            max_length=self.t5_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        res_batch = {k: v.to('cuda') for k, v in res_batch.items()}
        enc_text = self.t5_model(**res_batch).last_hidden_state
        mask = res_batch["attention_mask"]
        mask = ~mask.bool()
        return enc_text, mask


    def mask_text(self, text_list, force_mask=False):
        bs = len(text_list)
        dummy_text = "This is a null prompt with no semantic meaning."
        if force_mask:
            return [dummy_text for _ in range(bs)]
        elif self.training and self.label_drop_prob > 0.:
            text_list = list(text_list)
            random_mask = torch.rand(bs) < self.label_drop_prob
            for i, m in enumerate(random_mask):
                if m:
                    text_list[i] = dummy_text
            return tuple(text_list)
        else:
            return text_list


    def input_process(self, x):
        x = self.z_proj(x)
        x = self.query_pos_encoder(x)
        return x

    def forward_z(self, x, prompts, padding_mask, force_mask=False):
        prompts = self.mask_text(prompts, force_mask=force_mask)
        with torch.no_grad():
            cond_vector, text_mask = self.encode_text(prompts)
        cond_vector = self.cond_proj(cond_vector)
        cond_vector = self.text_aligner(cond_vector, text_mask)

        x = self.input_process(x)
        z = self.seqTransDecoder(tgt=x, memory=cond_vector, memory_key_padding_mask=text_mask,
                                 tgt_key_padding_mask=padding_mask)

        return z


    def forward(self, x, y, m_lens):
        bsz, seq_len, _ = x.size()
        m_lens = torch.div(m_lens, self.unit_length, rounding_mode='floor')
        gt_latents = x.clone().detach()

        rand_time = uniform((bsz,), device=x.device)
        rand_mask_probs = torch.cos(rand_time * math.pi * 0.5)
        num_masked = (seq_len * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((bsz, seq_len), device=x.device).argsort(dim=-1)
        mask = batch_randperm < num_masked.unsqueeze(-1)

        non_pad_mask = lengths_to_mask(m_lens, self.seq_len).to(x.device)
        mask = mask & non_pad_mask
        padding_mask = ~non_pad_mask

        mask_tokens = self.mask_token.repeat(bsz, seq_len, 1)

        x = torch.where(non_pad_mask.unsqueeze(-1), x, torch.zeros_like(x))
        x = torch.where(mask.unsqueeze(-1), mask_tokens, x)

        z = self.forward_z(x, y, padding_mask)

        target = gt_latents.reshape(bsz*seq_len, -1).repeat(self.flow_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.flow_batch_mul, 1)

        mask = mask.reshape(bsz*seq_len).repeat(self.flow_batch_mul)

        loss, loss_dict = self.flow_loss(z=z, target=target, mask=mask)

        return loss, loss_dict

    def forward_with_cfg(self, x, y, mask, key_padding_mask, cfg=4.0, temperature=1.0):

        z = self.forward_z(x, y, key_padding_mask, force_mask=False)

        aux_z = self.forward_z(x, y, key_padding_mask, force_mask=True)
        mixed_z = torch.cat([z, aux_z], dim=0)
        bsz, seq_len, embed_dim = mixed_z.size()

        mask = torch.cat([mask, mask], dim=0).reshape(bsz*seq_len)
        mixed_z = (mixed_z.reshape(bsz * seq_len, embed_dim))
        mixed_z = mixed_z[mask]

        sampled_token_latent = self.flow_loss.sample(mixed_z, cfg)
        sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
        mask, _ = mask.chunk(2, dim=0)
        x = x.reshape(bsz//2 * seq_len, self.token_embed_dim)
        x[mask.reshape(bsz//2 * seq_len)] = sampled_token_latent
        sampled_token_latent = x.reshape(bsz//2, seq_len, self.token_embed_dim)

        return sampled_token_latent


    def sample_tokens(self, bsz, m_lens, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, device='cuda', acc_ratio=1):
        m_lens = m_lens // self.unit_length
        seq_len = self.seq_len

        steps = int(seq_len//acc_ratio)

        key_padding_mask = ~lengths_to_mask(m_lens, seq_len).to(device)

        latents = torch.where(key_padding_mask.unsqueeze(-1), torch.zeros(bsz, seq_len, self.token_embed_dim).to(device),
                              self.mask_token.repeat(bsz, seq_len, 1))

        rand_vector = torch.rand_like(key_padding_mask, dtype=torch.float)

        masked_rand_schedule = torch.where(key_padding_mask, 1e5, rand_vector)

        for timestep in torch.linspace(0, 1, steps, device=device):
            rand_mask_prob = torch.cos(timestep * math.pi * 0.5)
            num_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_masked.unsqueeze(-1))

            mask_tokens = self.mask_token.repeat(bsz, seq_len, 1)
            latents_masked = torch.where(is_mask.unsqueeze(-1), mask_tokens, latents)

            if cfg_schedule == 'linear':
                cfg_iter = 1 + (cfg - 1) * timestep
            else:
                cfg_iter = cfg

            sampled_tokens = self.forward_with_cfg(latents_masked, labels, is_mask,
                                                   key_padding_mask=key_padding_mask, cfg=cfg_iter,
                                                   temperature=temperature)

            latents = torch.where(is_mask.unsqueeze(-1), sampled_tokens, latents_masked)

            no_mask = ~is_mask
            masked_rand_schedule = masked_rand_schedule.masked_fill(no_mask, 1e5) # preserve the denoised tokens from the last step

        latents = torch.where(key_padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents) # return to padding

        return latents


class TextAdapter(nn.Module):
    def __init__(self,
                 n_layers: int = 6,
                 d_model : int = 1024,
                 n_head  : int = 16,
                 ff_mult : int = 4,
                 dropout : float = 0.1):
        super().__init__()

        enc_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_head,
                        dim_feedforward=ff_mult * d_model,
                        dropout=dropout,
                        batch_first=True,
                        norm_first=True)

        self.blocks = nn.TransformerEncoder(
                          enc_layer,
                          num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states, key_padding_mask=None):
        x = self.blocks(hidden_states,
                        src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


def molingo_tiny(): # for quick local debug
    model = partial(MoLingo,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=4,
        flowloss_d=2, flowloss_w=256,
        mlp_ratio=2)
    return model


def molingo_base():
    model = partial(MoLingo,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        flowloss_d=6, flowloss_w=1024,
        mlp_ratio=4)
    return model


def molingo_large():
    model = partial(MoLingo,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        flowloss_d=8, flowloss_w=1280,
        mlp_ratio=4)
    return model


def molingo_huge():
    model = partial(MoLingo,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        flowloss_d=12, flowloss_w=1536,
        mlp_ratio=4)
    return model


def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask #(b, len)