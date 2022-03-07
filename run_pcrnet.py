import os
import torch
import argparse
import open3d as o3d
from torch.utils.data import DataLoader

from models import PointNet
from models import iPCRNet
from data import RegistrationData, ModelNet40Data, DudEData

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp_ipcrnet', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset_path', type=str, default='ModelNet40',
                        metavar='PATH', help='path to the input dataset')  # like '/path/to/ModelNet40'

    # settings for input data
    parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--num_points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')

    # settings for PointNet
    parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
                        help='train pointnet (default: tune)')
    parser.add_argument('--emb_dims', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

    # settings for on training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--pretrained_ptnet', default='', type=str,
                        metavar='PATH', help='path to pretrained ptnet file (default: null (no-use))')
    parser.add_argument('--pretrained_model', default='', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                                            metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args()
    return args


def tensor2ply(args, template, source, transformed_source, dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    if template.dim() == 3 and source.dim() == 3 and transformed_source.dim() == 3:

        for i, (t, s, transed_s) in enumerate(zip(template, source, transformed_source)):
            s = s.to('cpu').detach().numpy().copy()
            t = t.to('cpu').detach().numpy().copy()
            transed_s = transed_s.to('cpu').detach().numpy().copy()

            s_d = o3d.geometry.PointCloud()
            s_d.points = o3d.utility.Vector3dVector(s)
            s_d.paint_uniform_color([0, 0.651, 0.929])

            t_d = o3d.geometry.PointCloud()
            t_d.points = o3d.utility.Vector3dVector(t)
            t_d.paint_uniform_color([1, 0.706, 0])

            transed_s_d = o3d.geometry.PointCloud()
            transed_s_d.points = o3d.utility.Vector3dVector(transed_s)
            transed_s_d.paint_uniform_color([0, 0.706, 0])

            file_path = os.path.join(dir_path, f'{args.data.dataset_type}_{i}.ply')
            o3d.io.write_point_cloud(file_path, s_d + t_d + transed_s_d)


@hydra.main(config_path='config', config_name='default_run')
def main(args: DictConfig):

    # load models
    ptnet = PointNet(emb_dims=args.pointnet.emb_dims)
    model = iPCRNet(feature_model=ptnet)
    pretrained = os.path.join(os.getcwd(), 'checkpoints/models/best_model.t7')
    model.load_state_dict(
        torch.load(pretrained, map_location="cpu")
    )
    # model = model.to(args.device)
    model.eval()

    # load dataset
    if args.data.dataset_type == 'modelnet':
        testset = RegistrationData('PCRNet', ModelNet40Data(train=False, download=False, dir_path=to_absolute_path('data')))
    elif args.data.dataset_type == 'dude':
        testset = RegistrationData('PCRNet', DudEData(train=True))
    test_loader = DataLoader(testset, batch_size=args.training.batch_size, shuffle=False, drop_last=False, num_workers=args.training.workers)

    # get model output
    itr = iter(test_loader)
    output_dir_path = os.path.join(os.getcwd(), 'output_exsample')
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    for i in range(1):
        template, source, igt = next(itr)
        output = model(template, source)

        tensor2ply(args, template, source, output['transformed_source'], os.path.join(output_dir_path, f'batch{i}'))


if __name__ == '__main__':
    main()
