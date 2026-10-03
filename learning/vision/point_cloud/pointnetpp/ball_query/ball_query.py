import torch


def square_distance(query_coords, reference_coords):
    """
    计算两组点之间的欧氏距离平方。

    query_coords: [B, S, 3]
    reference_coords: [B, N, 3]

    return:
        squared_distances: [B, S, N]
    """
    batch_size, num_queries, _ = query_coords.shape
    _, num_reference_points, _ = reference_coords.shape

    squared_distances = -2 * torch.matmul(
        query_coords,
        reference_coords.transpose(1, 2)
    )
    squared_distances += torch.sum(query_coords ** 2, dim=-1).view(
        batch_size,
        num_queries,
        1
    )
    squared_distances += torch.sum(reference_coords ** 2, dim=-1).view(
        batch_size,
        1,
        num_reference_points
    )

    return squared_distances


def query_ball_point(radius, max_neighbors, point_coords, centroid_coords):
    """
    为每个中心点查询半径范围内的邻域点索引。

    point_coords: [B, N, 3]
    centroid_coords: [B, S, 3]

    return:
        neighbor_indices: [B, S, max_neighbors]
    """
    device = point_coords.device
    batch_size, num_points, _ = point_coords.shape
    _, num_centroids, _ = centroid_coords.shape

    neighbor_indices = torch.arange(
        num_points,
        dtype=torch.long,
        device=device
    )
    neighbor_indices = neighbor_indices.view(1, 1, num_points).repeat(
        batch_size,
        num_centroids,
        1
    )

    squared_distances = square_distance(centroid_coords, point_coords)
    neighbor_indices[squared_distances > radius ** 2] = num_points
    neighbor_indices = neighbor_indices.sort(dim=-1)[0]
    neighbor_indices = neighbor_indices[:, :, :max_neighbors]

    first_neighbor_indices = neighbor_indices[:, :, 0]
    first_neighbor_indices = first_neighbor_indices.view(
        batch_size,
        num_centroids,
        1
    ).repeat(1, 1, max_neighbors)

    invalid_mask = neighbor_indices == num_points
    neighbor_indices[invalid_mask] = first_neighbor_indices[invalid_mask]

    return neighbor_indices
