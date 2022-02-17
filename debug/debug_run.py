import pathlib
import sys
module_path = pathlib.Path(__file__, "..", '..').resolve()
if module_path not in sys.path:
    sys.path.append(str(module_path))
print(module_path)

import os
import open3d as o3d

from data import DudEData, RegistrationData


def tensor2ply(template, source, dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    if template.dim() == 3 and source.dim() == 3:

        for i, (t, s) in enumerate(zip(template, source)):
            s = s.to('cpu').detach().numpy().copy()
            t = t.to('cpu').detach().numpy().copy()

            s_d = o3d.geometry.PointCloud()
            s_d.points = o3d.utility.Vector3dVector(s)
            s_d.paint_uniform_color([0, 0.651, 0.929])

            t_d = o3d.geometry.PointCloud()
            t_d.points = o3d.utility.Vector3dVector(t)
            t_d.paint_uniform_color([1, 0.706, 0])

            file_path = os.path.join(dir_path, f'modelnet_{i}.ply')
            o3d.io.write_point_cloud(file_path, s_d + t_d)

    else:

        s = source
        t = template

        s = s.to('cpu').detach().numpy().copy()
        t = t.to('cpu').detach().numpy().copy()

        s_d = o3d.geometry.PointCloud()
        s_d.points = o3d.utility.Vector3dVector(s)
        s_d.paint_uniform_color([0, 0.651, 0.929])

        t_d = o3d.geometry.PointCloud()
        t_d.points = o3d.utility.Vector3dVector(t)
        t_d.paint_uniform_color([1, 0.706, 0])

        file_path = os.path.join(dir_path, 'set.ply')
        o3d.io.write_point_cloud(file_path, s_d + t_d)


if __name__ == '__main__':
    dataset = RegistrationData('PCRNet', DudEData(train=False))
    s, t, _ = dataset[0]
    tensor2ply(s, t, 'logs/dataloader_check')
