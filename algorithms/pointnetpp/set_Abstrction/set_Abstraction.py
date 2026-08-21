from mix_fps_ballQuery import mix_fps_ballQuery

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetAbstraction(nn.Module):

    def __init__(
            self,
            num_centroid,
            radius,
            max_neighbors,
            in_channel,
            mlp):

                super().__init__()

                self.num_centroid = num_centroid
                self.radius = radius
                self.max_neighbors = max_neighbors

                self.mlp_convs = nn.ModuleList()
                self.mlp_bns = nn.ModuleList()

                last_channel = in_channel

                for out_channel in mlp:

                        self.mlp_convs.append(
                                nn.Conv2d(
                                        last_channel,
                                        out_channel,
                                        kernel_size=1
                                )
                        )

                        self.mlp_bns.append(
                                nn.BatchNorm3d(
                                        out_channel
                                )
                        )

                        last_channel = out_channel

    def forward(self, xyz, points):
            """"
            xyz:
                [B, N, 3]

            points:
                [B, N, D]
            """

            #fps + BallQuery


            new_xyz, new_points = mix_fps_ballQuery.sample_and_group(
                    self.num_centroid,
                    self.radius,
                    self.max_neighbors,
                    xyz,
                    points
            )

            #new_points
            #[B, S, N, C]
            #Conv2d需要
            #[B, C, K, S]

            new_points = new_points.permute(
                    0, 3, 2, 1
            )

            #局部MLP
            for conv, bn in zip(
                    self.mlp_convs,
                    self.mlp_bns
            ):
                new_points = F.relu(
                       bn(
                              conv(new_points)
                       )
                )

            #局部 Max Pool
            new_points = torch.max(
                   new_points,
                   dim=2
            )[0]

            #[B, C, S]
            #|
            #V
            #[B, S, C]

            new_points = new_points.permute(
                   0, 2, 1
            )

            return new_xyz, new_points