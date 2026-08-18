import torch

def index_points(points, idx):
    """
    points:[B, N, C]

    idx:[B, S, C]

    return:
        [B, S, C]
        或
        [B, S, K, C]
    """

    device = points.device

    view_shape = list(idx.shape)

    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)

    repeat_shape[0] = 1

    B = points.shape[0]

    batch_indices = torch.arange(
        B,
        dtype=torch.long,
        device=device
    ).view(view_shape).repeat(repeat_shape)

    new_points = points[
        batch_indices,
        idx,
        :
    ]

    return new_points



