import sys
sys.path.append('.')

import os
from os.path import join as pjoin

import torch
from mogen.utils.collate import collate_x_dict_with_padding, collate_x_dict
from mogen.models.tmr.load_tmr import load_tmr_model


def build_models():
    tmr_root = 'mogen/checkpoints/TMR'
    tmr_model, text_to_token_emb, normalizer = load_tmr_model(nfeats=263)
    pt_path = pjoin(tmr_root, 'models/tmr_humanml3d_guoh3dfeats/last_weights')

    for fname in os.listdir(pt_path):
        module_name, ext = os.path.splitext(fname)

        if ext != ".pt":
            continue

        module = getattr(tmr_model, module_name, None)
        if module is None:
            continue

        module_path = os.path.join(pt_path, fname)
        state_dict = torch.load(module_path)
        module.load_state_dict(state_dict)
        # logger.info(f"    {module_name} loaded")
        print((f"    {module_name} loaded"))

    return tmr_model, text_to_token_emb, normalizer

class TMREvaluatorModelWrapper(object):

    def __init__(self, device):

        self.tmr_model, self.text_to_token_emb, self.normalizer = build_models()
        self.tmr_model.to(device)
        self.tmr_model.eval()
        self.device = device

    # Please note that the results does not follow the order of inputs
    def get_co_embeddings(self, motions, m_lens, clip_text=None):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            motions = self.normalizer(motions).float()
            motion_x_dict = collate_x_dict_with_padding(motions, m_lens)
            motion_embedding = self.tmr_model.encode(motion_x_dict, sample_mean=True)

            text_emb = self.text_to_token_emb(clip_text)
            text_x_dict = collate_x_dict(text_emb, device=self.device)
            text_embedding = self.tmr_model.encode(text_x_dict, sample_mean=True)

        return text_embedding, motion_embedding

    # Please note that the results does not follow the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            motions = self.normalizer(motions).float()
            motion_x_dict = collate_x_dict_with_padding(motions, m_lens)
            motion_embedding = self.tmr_model.encode(motion_x_dict, sample_mean=True)
        return motion_embedding