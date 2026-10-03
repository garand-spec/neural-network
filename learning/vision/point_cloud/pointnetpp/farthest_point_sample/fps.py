import torch


def farthest_point_sample(point_coords, num_centroids):
    """
    从点云中采样距离已选点集最远的点。

    point_coords: [B, N, 3]
    num_centroids: 采样中心点数

    return:
        centroid_indices: [B, num_centroids]
    """
    batch_size, num_points, _ = point_coords.shape

    centroid_indices = torch.zeros(
        batch_size,
        num_centroids,
        dtype=torch.long,
        device=point_coords.device
    )
    minimum_squared_distances = torch.ones(
        batch_size,
        num_points,
        device=point_coords.device
    ) * 1e10

    farthest_indices = torch.randint(
        0,
        num_points,
        (batch_size,),
        dtype=torch.long,
        device=point_coords.device
    )
    batch_indices = torch.arange(
        batch_size,
        dtype=torch.long,
        device=point_coords.device
    )

    for centroid_index in range(num_centroids):
        centroid_indices[:, centroid_index] = farthest_indices
        current_centroid_coords = point_coords[
            batch_indices,
            farthest_indices,
            :
        ].view(batch_size, 1, 3)

        squared_distances = torch.sum(
            (point_coords - current_centroid_coords) ** 2,
            dim=-1
        )
        closer_mask = squared_distances < minimum_squared_distances
        minimum_squared_distances[closer_mask] = squared_distances[closer_mask]
        farthest_indices = torch.max(
            minimum_squared_distances,
            dim=-1
        )[1]

    return centroid_indices
