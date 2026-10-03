import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure sibling modules in the parent package directory are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from set_Abstraction import set_Abstraction
#双层PointNet++网络
class PointNet2Classifier(nn.Module):

    def __init__(self, num_classes=40):
        super().__init__()

        #第一层
        self.sa1 = set_Abstraction.PointNetAbstraction(
            num_centroid=512,
            radius=0.2,
            max_neighbors=32,
            in_channel=3,
            mlp=[64, 64, 128]
        )

        #第二层
        #输入： xyz相对坐标 3 + 上一层特征128 = 131

        self.sa2 = set_Abstraction.PointNetAbstraction(
            num_centroid=128,
            radius=0.4,
            max_neighbors=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256]
        )

        self.fc1 = nn.Linear(256, 256)

        self.bn1 = nn.BatchNorm1d(
            256
        )

        self.drop1 = nn.Dropout(
            0.4
        )

        self.fc2 = nn.Linear(
            256,
            num_classes
        )

    def forward(self, xyz):
        
        # xyz
        #
        # [B,1024,3]

        l1_xyz, l1_points = self.sa1(
            xyz,
            None
        )

        # l1_xyz
        # [B,512,3]
        #
        # l1_points
        # [B,512,128]

        l2_xyz, l2_points = self.sa2(
            l1_xyz,
            l1_points
        )

        # l2_xyz
        # [B,128,3]
        #
        # l2_points
        # [B,128,256]

        #[B, 256]

        # Global feature: max-pool across point dimension (N)
        x = torch.max(l2_points, dim=1)[0]

        x = self.drop1(
            F.relu(
                self.bn1(self.fc1(x))
            )
        )

        x = self.fc2(x)

        return x

if __name__ == "__main__":
    xyz = torch.randn(
        8,
        1024,
        3
    )

    model = PointNet2Classifier(
        num_classes=10
    )

    output = model(xyz)

    print(output.shape)