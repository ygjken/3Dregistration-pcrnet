import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import os
import json

from hydra.utils import to_absolute_path


class DudESourceData(Dataset):
    def __init__(
        self,
        train=True,
        num_points=1024,
        sampler='random',
        data_path=to_absolute_path('data/dude'),
        do_transform=False
    ):
        super().__init__()
        self.path_prefix = data_path
        self.train = train

        json_path = os.path.join(self.path_prefix, 'list.json')
        json_file = open(json_path, 'r')
        self.namelist = json.load(json_file)

        self.pcs = []

        # read from .json and .ply
        for i, (_) in enumerate(self.namelist):
            pc = o3d.io.read_point_cloud(os.path.join(self.path_prefix, "ply", self.namelist[i]['source']) + '.ply')
            pc = self._random_sample(pc, num_points)

            self.pcs += [pc]

        # separate for train mode or test mode
        sep = int(len(self.pcs) * 0.7)
        if self.train:
            self.pcs = self.pcs[:sep]
        else:
            self.pcs = self.pcs[sep:]

        # numpy to tensor
        self._numpy_to_tensor()

        # transform
        if do_transform:
            self._transform()

    def _random_sample(self, pc: o3d.geometry.PointCloud, n: int):
        """run random sampling to a specified size

        Args:
            pc (o3d.geometry.PointCloud): Point cloud before sampling
            n (int): number of point after sampling

        Returns:
            np.array : sampled point cloud in numpy
        """
        pc_len = len(np.asarray(pc.points))
        pc = pc.random_down_sample((n + 0.1) / pc_len)
        return np.asarray(pc.points)

    def _numpy_to_tensor(self):
        for i, _ in enumerate(self.pcs):
            self.pcs[i] = torch.from_numpy(self.pcs[i].astype(np.float32)).clone()  # numpy to tensor

    def _transform(self, mean=0, var=0.1):
        for i, _ in enumerate(self.pcs):
            all_points = self.pcs[i]
            mu = torch.mean(all_points)
            sigma2 = torch.mean((all_points - mu) ** 2)

            self.pcs[i] = (self.pcs[i] - mu) / torch.sqrt(sigma2 / var) + mean

    def __len__(self):
        return len(self.pcs)

    def __getitem__(self, idx):
        return self.pcs[idx]
