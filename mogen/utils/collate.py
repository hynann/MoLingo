import torch

from typing import List, Dict, Optional
from torch import Tensor
from torch.utils.data import default_collate


def length_to_mask(length, device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask

def length_to_mask_with_padding(length, max_len, device):
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_x_dict(lst_x_dict: List, *, device: Optional[str] = None) -> Dict:
    x = collate_tensor_with_padding([x_dict["x"] for x_dict in lst_x_dict])
    if device is not None:
        x = x.to(device)
    length = [x_dict["length"] for x_dict in lst_x_dict]
    mask = length_to_mask(length, device=x.device)
    batch = {"x": x, "length": length, "mask": mask}
    return batch

def collate_x_dict_with_padding(x, m_lens):
    # motions [bs, max_len, 263], m_lens [bs,]
    max_len = x.size(1)
    mask = length_to_mask_with_padding(m_lens, max_len, x.device)
    batch = {"x": x, "length": m_lens, "mask": mask}
    return batch


def collate_text_motion(lst_elements: List, *, device: Optional[str] = None) -> Dict:
    one_el = lst_elements[0]
    keys = one_el.keys()

    x_dict_keys = [key for key in keys if "x_dict" in key]
    other_keys = [key for key in keys if "x_dict" not in key]

    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and device is not None:
            batch[key] = val.to(device)

    for key in x_dict_keys:
        batch[key] = collate_x_dict([x[key] for x in lst_elements], device=device)
    return batch


def collate_text_motion_multiple_texts(lst_elements: List, *, device: Optional[str] = None):
    other_keys = ['keyid', 'sent_emb']

    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    batch["text"] = [elt["text"] for elt in lst_elements]

    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and device is not None:
            batch[key] = val.to(device)

    batch["motion_x_dict"] = collate_x_dict([x["motion_x_dict"] for x in lst_elements], device=device)

    batch["text_slices"] = []
    current_index = 0
    for elt in lst_elements:
        batch["text_slices"].append((current_index, current_index + len(elt["text"])))
        current_index += len(elt["text"])

    texts_concat = [x_dict for x in lst_elements for x_dict in x["text_x_dict"]]
    batch["text_x_dict"] = collate_x_dict(
        texts_concat,
        device=device
    )
    return batch
