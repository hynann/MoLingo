import torch
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def match_dim_with_data(
    t: torch.Tensor | float | list[float],
    x_shape: tuple,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    expand_dim: bool = True,
):
    r"""
    Format the time tensor `t` to match the batch size and dimensions of the data.

    This function ensures that the time tensor `t` is properly formatted to match the batch size specified by `x_shape`.
    It handles various input types for `t`, including scalars, lists, or tensors, and converts `t` into a tensor with
    appropriate shape, device, and dtype. Optionally, it can expand `t` to match the data dimensions beyond the batch size.

    Args:
        t (`torch.Tensor`, `float`, or `list[float]`):
            The time(s) to be matched with the data dimensions. Can be a scalar, a list of floats, or a tensor.
        x_shape (`tuple`):
            The shape of the data tensor, typically `(batch_size, ...)`.
        device (`torch.device`, optional, defaults to `torch.device("cpu")`):
            The device on which to place the time tensor.
        dtype (`torch.dtype`, optional, defaults to `torch.float32`):
            The data type of the time tensor.
        expand_dim (`bool`, optional, defaults to `True`):
            Whether to expand `t` to match the dimensions after the batch dimension.

    Returns:
        t_reshaped (`torch.Tensor`):
            The time tensor `t`, formatted to match the batch size or dimensions of the data.

    Example:
        ```python
        >>> x_shape = (16, 3, 32, 32)
        >>> t_prepared = match_dim_with_data([0.5], x_shape, expand_dim=True)
        >>> t_prepared.shape
        torch.Size([16, 1, 1, 1])

        >>> x_shape = (16, 3, 32, 32)
        >>> t_prepared = match_dim_with_data([0.5], x_shape, expand_dim=False)
        >>> t_prepared.shape
        torch.Size([16])
        ```
    """
    B = x_shape[0]  # Batch size
    ndim = len(x_shape)

    if isinstance(t, float):
        # Create a tensor of shape (B,) with the scalar value
        t = torch.full((B,), t, device=device, dtype=dtype)
    elif isinstance(t, list):
        if len(t) == 1:
            # If t is a list of length 1, repeat the scalar value B times
            t = torch.full((B,), t[0], device=device, dtype=dtype)
        elif len(t) == B:
            t = torch.tensor(t, device=device, dtype=dtype)
        else:
            raise ValueError(
                f"Length of t list ({len(t)}) does not match batch size ({B}) and is not 1."
            )
    elif isinstance(t, torch.Tensor):
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 0:
            # Scalar tensor, expand to (B,)
            t = t.repeat(B)
        elif t.ndim == 1:
            if t.shape[0] == 1:
                # Tensor of shape (1,), repeat to (B,)
                t = t.repeat(B)
            elif t.shape[0] == B:
                # t is already of shape (B,)
                pass
            else:
                raise ValueError(
                    f"Batch size of t ({t.shape[0]}) does not match x ({B})."
                )
        elif t.ndim == 2:
            if t.shape == (B, 1):
                # t is of shape (B, 1), squeeze last dimension
                t = t.squeeze(1)
            elif t.shape == (1, 1):
                # t is of shape (1, 1), expand to (B,)
                t = t.squeeze().repeat(B)
            else:
                raise ValueError(
                    f"t must be of shape ({B}, 1) or (1, 1), but got {t.shape}"
                )
        else:
            raise ValueError(f"t can have at most 2 dimensions, but got {t.ndim}")
    else:
        raise TypeError(
            f"t must be a torch.Tensor, float, or a list of floats, but got {type(t)}."
        )

    # Reshape t to have singleton dimensions matching x_shape after the batch dimension
    if expand_dim:
        expanded_dims = [1] * (ndim - 1)
        t = t.view(B, *expanded_dims)

    return t
