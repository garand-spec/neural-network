import torch

def square_distance(src, dst):
    """
    计算两组点之间的欧氏距离平方

    src[B, S, 3]
    dst[B, N, 3]

    return:
        dist [B, S, N]
    """

    B, S, _ = src.shape
    _, N, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1).view(B, S, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, N)

    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball Query

    radius:
        球形邻域半径

    nsample:
        每个球最多选择点数
    
    xyz:
        原始点云
        [B, N, 3]

    new_xyz:
        中心点，一般最远点采样后获得
        [B, S, 3] 
    
    return:
        group_idx:
        [B, S, nsample]

        保存每个中心点附近点的下标
    """

    device = xyz.device

    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    #给所有原始点建立索引
    #[0, 1, 2, 3, 4, 5......, N-1]
    #扩展为[B, S, N]

    group_idx = torch.arange(
        N,
        dtype=torch.long,
        device=device
    )

    group_idx = group_idx.view(1, 1, N).repeat(B, S, 1)

    #计算每个中心点到原始点的距离
    #new_xyz: [B, S ,3]
    #xyz: [B, N, 3]
    #sqdists: [B, S, N]

    sqrdists = square_distance(new_xyz, xyz)

    #把球外面的点标记成N
    #如果：distance > radius
    #这个点就不属于这个中心的邻域

    group_idx[sqrdists > radius ** 2] = N

    #排序
    #合法点： 0~N-1
    #非法点：N
    #所以排序以后
    #[合法点, 合法点, ...., N, N]

    group_idx = group_idx.sort(dim=-1)[0]

    #5.最多只保留nsample 个点
    
    group_idx = group_idx[:, :, :nsample]

    #如果球里面的点不足nsample个,用第一个合法的点补齐
    group_first = group_idx[:, :, 0]

    group_first = group_first.view(B, S, 1).repeat(1, 1, nsample)

    mask = group_idx == N

    group_idx[mask] = group_first[mask]

    return group_idx




