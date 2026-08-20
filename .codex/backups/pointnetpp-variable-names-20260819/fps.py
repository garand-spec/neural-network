import torch

def farthest_point_sample(xyz, npoint):
    """
    xyz: [B, N, 3]
    B: batch size
    N: 点的数量
    3: x, y, z坐标

    npoint: 
        要采样的点数

    return:
        centroid: [B, npoint]
        返回采样点在原始点云的下表
    """

    B, N, C = xyz.shape

    #保存最终选出来的点的下标
    centroids = torch.zeros(
        B, npoint,
        dtype=torch.long,
        device=xyz.device
    )

    #distance[b][j]
    #表现第j个点到“当前已选点集合”的最近距离
    distance = torch.ones(
        B, N,
        device=xyz.deivice
    ) * 1e10

    #随机选择第一个点
    farthest = torch.randint(
        0, N,
        (B,),
        dtype=torch.long,
        device=xyz.device
    )

    #batch 下标
    batch_indices = torch.arange(
        B,
        dtype=torch.long,
        device=xyz.device
    )

    for i in range(npoint):

        #1. 保存当前最远点
        centroids[:, i] = farthest

        #2.取出这个点的坐标
        centroid = xyz[
            batch_indices,
            farthest,
            :
        ].view(B, 1, 3)

        #3.计算所有点到当前centroids的欧氏距离平方
        dist = torch.sum(
            (xyz - centroid) ** 2,
            dim=-1 
            )

        #dist
        #[B, N]

        #4. 更新每个点到“已选点集合”的最近距离
        mask = dist < distance

        distance[mask] = dist[mask]

        #5.找最近距离最大的点
        #作为下一轮的farthest
        farthest = torch.max(
            distance,
            dim=-1
        )[1]

    return centroids