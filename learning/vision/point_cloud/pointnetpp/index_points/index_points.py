import torch


def index_points(point_data, point_indices):
    """
    按索引从点数据中取值。

    point_data: [B, N, C]
    point_indices: [B, S] 或 [B, S, K]

    return:
        indexed_points: [B, S, C] 或 [B, S, K, C]
    """
    device = point_data.device
    batch_size = point_data.shape[0]

    batch_index_shape = list(point_indices.shape)
    batch_index_shape[1:] = [1] * (len(batch_index_shape) - 1)

    batch_index_repeats = list(point_indices.shape)
    batch_index_repeats[0] = 1

    batch_indices = torch.arange(
        batch_size,
        dtype=torch.long,
        device=device
    ).view(batch_index_shape).repeat(batch_index_repeats)

    indexed_points = point_data[
        batch_indices,
        point_indices,
        :
    ]

    return indexed_points
