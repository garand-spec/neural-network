from farthest_point_sample import fps
from ball_query import ball_query
from index_points import index_points

def sample_and_group(npoint, radius, nsample, xyz, points):

    """
    xyz: [B, N, 3]
    
    points: [B, N, D]
    
    return:
        new_xyz: [B, npoint, 3]
        
        new_points: [B, npoint, nsample, 3+D]    
    """

    #FPS
    fps_idx = fps.farthest_point_sample(
        xyz,
        npoint
    )

    #中心点坐标
    new_xyz = index_points.index_points(
        xyz,
        fps_idx
    )

    #BallQuery
    idx = ball_query.query_ball_point(
        radius,
        nsample,
        xyz,
        new_xyz
    )

    #取领域点坐标
    grouped_xyz = index_points.index_points(
        xyz,
        idx
    )

    #转相对坐标
    grouped_xyz_norm = (
        grouped_xyz -
        new_xyz.view(
            new_xyz.shape[0],
            npoint,
            1,
            3
        )
    )