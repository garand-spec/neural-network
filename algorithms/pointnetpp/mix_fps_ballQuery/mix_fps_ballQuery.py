from farthest_point_sample import fps
from ball_query import ball_query
from index_points import index_points

import pytorch as torch

def sample_and_group(
    num_centroids,
    radius,
    max_neighbors,
    point_coords,
    point_features
):
    """
    对点云进行最远点采样，并查询每个中心点的球形邻域。

    point_coords: [B, N, 3]
    point_features: [B, N, D]

    return:
        centroid_coords: [B, num_centroids, 3]
        grouped_points: [B, num_centroids, max_neighbors, 3 + D]
    """
    #中心点索引
    sampled_point_indices = fps.farthest_point_sample(
        point_coords,
        num_centroids
    )
    #采样中心点坐标
    centroid_coords = index_points.index_points(
        point_coords,
        sampled_point_indices
    )
    #邻域点索引
    neighbor_indices = ball_query.query_ball_point(
        radius,
        max_neighbors,
        point_coords,
        centroid_coords
    )
    #取邻域点坐标
    grouped_coords = index_points.index_points(
        point_coords,
        neighbor_indices
    )
    #转为相对坐标
    normalized_grouped_coords = (
        grouped_coords - centroid_coords.view(
        centroid_coords.shape[0],
        num_centroids,
        1,
        3
        )
    )

    #如果本来就有一些特征

    if point_features is not None:

        grouped_feature = index_points.index_points(
            point_features,
            neighbor_indices
        )

        centroid_feature = torch.cat(
            [
                normalized_grouped_coords,
                grouped_feature
            ],
            dim=-1
        )

    else:

        centroid_feature = normalized_grouped_coords

    return centroid_coords, centroid_feature