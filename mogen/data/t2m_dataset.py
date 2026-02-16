import sys
sys.path.append('.')

import random
import numpy as np
import codecs as cs
import torch

from os.path import join as pjoin
from torch.utils import data
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate

from mogen.utils.motion_representation import recover_from_ric

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

# debug = True
debug = False

class Text2MotionDatasetEval(data.Dataset):
    def __init__(self, opt, mean, std, mean_eval, std_eval, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        self.njoints = 22

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                id_list.append(line)
        if debug:
            id_list = id_list[:100]

        new_name_list = []
        hml_key_list = []
        length_list = []

        for name in tqdm(id_list):
            try:
                motion_path = pjoin(opt.motion_dir, name + '.npy')
                motion = np.load(motion_path)
                if (len(motion)) < min_motion_len or (len(motion) > 200):
                    continue
                # if len(motion) > self.max_motion_length:
                #     motion = motion[:self.max_motion_length]
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) > 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                hml_key_list.append(name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    hml_key_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list, hml_key_list = zip(*sorted(zip(new_name_list, length_list, hml_key_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.mean_eval = mean_eval
        self.std_eval = std_eval
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.hml_key_list = hml_key_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def inv_transform_torch(self, data):
        mean = torch.tensor(self.mean).to(data)
        std = torch.tensor(self.std).to(data)
        return data * std + mean

    def feats_to_eval_mardm(self, data):
        mean = torch.tensor(self.mean).to(data)
        std = torch.tensor(self.std).to(data)
        mean_eval = torch.tensor(self.mean_eval).to(data)
        std_eval = torch.tensor(self.std_eval).to(data)

        data_rst = data * std + mean
        data_rst = data_rst[..., :67]
        data_norm = (data_rst - mean_eval) / std_eval
        return data_norm

    def feats_denorm(self, data):
        mean = torch.tensor(self.mean).to(data)
        std = torch.tensor(self.std).to(data)
        data_rst = data * std + mean
        return data_rst

    def feats2joints(self, features: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean).to(features)
        std = torch.tensor(self.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def feats_to_eval(self, data):
        mean = torch.tensor(self.mean).to(data)
        std = torch.tensor(self.std).to(data)
        mean_eval = torch.tensor(self.mean_eval).to(data)
        std_eval = torch.tensor(self.std_eval).to(data)

        data_rst = data * std + mean
        data_norm = (data_rst - mean_eval) / std_eval
        return data_norm

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        hml_key = self.hml_key_list[idx]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), hml_key