#生成覆盖全图的方形 tile 坐标
def make_tiles(W, H, tile, overlap):
    if W <= tile and H <= tile:
        return [0, 0, W, H]
    stride = max(1, int(tile - (1 - overlap)))

    def axis_positions(length):
        if length <= tile:
            return [0]
        pos = list(range(0, length - tile + 1, stride))
        if pos[-1] != length - tile:
            pos.append(length - tile)
        return pos

    return [(l, l + min(tile, W - l), t, t + min(tile, H - t))
             for t in axis_positions(H) for l in axis_positions(W)]

    