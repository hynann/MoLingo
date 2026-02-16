import sys
sys.path.append('.')

import os
import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from mogen.utils.word_vectorizer import POS_enumerator
import clip
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#################################################################################
#                                    Evaluators                                 #
#################################################################################
def build_evaluators(dim_pose, dataset_name, dim_movement_enc_hidden, dim_movement_latent, dim_word, dim_pos_ohot, dim_text_hidden,
                     dim_coemb_hidden, dim_motion_hidden, checkpoints_dir, device):
    movement_enc = MovementConvEncoder(dim_pose, dim_movement_enc_hidden, dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=dim_word,
                                  pos_size=dim_pos_ohot,
                                  hidden_size=dim_text_hidden,
                                  output_size=dim_coemb_hidden,
                                  device=device)

    motion_enc = MotionEncoderBiGRUCo(input_size=dim_movement_latent,
                                      hidden_size=dim_motion_hidden,
                                      output_size=dim_coemb_hidden,
                                      device=device)
    contrast_model = MotionCLIP(dim_pose)

    checkpoint = torch.load(os.path.join(checkpoints_dir, dataset_name, 'text_mot_match_mardm', 'model', 'finest.tar'),
                            map_location=device)
    checkpoint_clip = torch.load(os.path.join(checkpoints_dir, dataset_name, 'text_mot_match_clip', 'model', 'finest.tar'),
                            map_location=device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    contrast_model.load_state_dict(checkpoint_clip['contrast_model'])
    print('Loading Evaluators')
    return text_enc, motion_enc, movement_enc, contrast_model

class MARDMEvaluators(object):

    def __init__(self, dataset_name, device):
        if dataset_name == 't2m':
            dim_pose = 67
        elif dataset_name == 'kit':
            dim_pose = 64
        else:
            raise KeyError('Dataset not Recognized!!!')

        dim_word = 300
        dim_pos_ohot = len(POS_enumerator)
        dim_motion_hidden = 1024
        dim_movement_enc_hidden = 512
        dim_movement_latent = 512
        dim_text_hidden = 512
        dim_coemb_hidden = 512
        checkpoints_dir = 'mogen/checkpoints'
        self.unit_length = 4

        self.text_encoder, self.motion_encoder, self.movement_encoder, self.contrast_model \
        = build_evaluators(dim_pose, dataset_name, dim_movement_enc_hidden, dim_movement_latent, dim_word,
                            dim_pos_ohot, dim_text_hidden, dim_coemb_hidden, dim_motion_hidden, checkpoints_dir, device)
        self.device = device

        self.text_encoder.to(device)
        self.motion_encoder.to(device)
        self.movement_encoder.to(device)
        self.contrast_model.to(device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()
        self.contrast_model.eval()

    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, captions, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            '''clip based'''
            clip_em = self.contrast_model.encode_motion(motions.clone(), m_lens)
            clip_et = self.contrast_model.encode_text(captions)
            clip_em = clip_em / clip_em.norm(dim=1, keepdim=True)
            clip_et = clip_et / clip_et.norm(dim=1, keepdim=True)

            '''original architecture'''
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return (text_embedding, motion_embedding), (clip_et, clip_em)

    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            '''clip based'''
            clip_em = self.contrast_model.encode_motion(motions.clone(), m_lens)
            clip_em = clip_em / clip_em.norm(dim=1, keepdim=True)

            '''original architecture'''
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding, clip_em

#################################################################################
#                                 Inner Architectures                           #
#################################################################################
def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        return self.pe[pos]


class PositionalEncodingCLIP(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncodingCLIP, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


def no_grad(nets):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = False

def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class MovementConvDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)

class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim, ff_size, num_layers, num_heads, dropout, activation):
        super().__init__()
        self.input_feats = in_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncodingCLIP(self.latent_dim, self.dropout, max_len=2000)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)


    def forward(self, motion, padding_mask):
        B, T, D  = motion.shape

        x_emb = self.embed_motion(motion)

        emb = torch.cat([self.query_token[torch.zeros(B, dtype=torch.long, device=motion.device)][:,None], x_emb], dim=1)

        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = h.permute(1, 0, 2)
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        h = h.permute(1, 0, 2)
        h = self.out_ln(h)
        motion_emb = self.out(h[:,0])

        return motion_emb


class MotionCLIP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.motion_encoder = MotionEncoder(in_dim, 512, 1024, 8, 8, 0.2, 'gelu')

        clip_model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        no_grad(self.token_embedding)

        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.2,
            activation="gelu",)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=8)
        self.text_ln = nn.LayerNorm(512)
        self.out = nn.Linear(512, 512)

    def encode_motion(self, motion, m_lens):
        seq_len = motion.shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_embedding = self.motion_encoder(motion, padding_mask.to(motion.device))
        return motion_embedding

    def encode_text(self, text):
        device = next(self.parameters()).device

        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.token_embedding(text).float()
            pe_tokens = x + self.positional_embedding.float()
        pe_tokens = pe_tokens.permute(1,0,2)
        out = self.textTransEncoder(pe_tokens)
        out = out.permute(1, 0, 2)
        out = self.text_ln(out)

        out = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        out = self.out(out)
        return out

    def forward(self, motion, m_lens, text):
        motion_features = self.encode_motion(motion, m_lens)
        text_features = self.encode_text(text)

        motion_features = motion_features / motion_features .norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_motion = logit_scale * motion_features @ text_features.t()
        logits_per_text = logits_per_motion.t()
        return logits_per_motion, logits_per_text

    def forward_loss(self, motion, m_lens, text):
        logits_per_motion, logits_per_text = self.forward(motion, m_lens, text)
        labels = torch.arange(len(logits_per_motion)).to(logits_per_motion.device)

        image_loss = F.cross_entropy(logits_per_motion, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        return loss
