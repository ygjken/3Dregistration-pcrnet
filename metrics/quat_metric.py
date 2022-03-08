import torch


class QuatMetric:
    def __init__(self):
        pass

    def compute_errors(self, v1, v2):
        rot_err = torch.mean(2 * torch.acos(2 * (torch.sum(self._quat(v1) * self._quat(v2), dim=1)) ** 2 - 1))
        trans_err = torch.mean(torch.sqrt((self._trans(v1) - self._trans(v2)) ** 2))
        return rot_err, trans_err

    def _quat(self, v):
        return v[:, 0:4]

    def _trans(self, v):
        return v[:, 4:]
