import argparse
import os
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
    sys.path.append(os.path.join(BASE_DIR, os.pardir))
    os.chdir(os.path.join(BASE_DIR, os.pardir))

from models import PointNet
from models import iPCRNet
from losses import ChamferDistanceLoss, EarthMoverDistanceFunction, earth_mover_distance
from data import RegistrationData, ModelNet40Data, DudEData, DudESourceData
from metrics import QuatMetric


def _init_(args):
    if not os.path.exists('checkpoints/' + 'models'):
        os.makedirs('checkpoints/' + 'models')


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def test_one_epoch(device, model, test_loader, loss):
    model.eval()
    test_loss = 0.0
    pred = 0.0
    quat_error = 0.0
    count = 0
    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt = data

        template = template.to(device)
        source = source.to(device)
        igt = igt.view(-1, 7).to(device)

        output = model(template, source)
        if loss == 'cd':
            loss_val = ChamferDistanceLoss()(template, output['transformed_source'])
        elif loss == 'emd':
            loss_val = earth_mover_distance(template, output['transformed_source'])

        quat_rot, quat_trans = QuatMetric().compute_errors(igt, output['pose_7d'])
        quat_error += (quat_rot + quat_trans).item()
        test_loss += loss_val.item()
        count += 1

    test_loss = float(test_loss) / count
    quat_error = float(quat_error) / count
    return test_loss, quat_error


def test(args, model, test_loader, textio, device):
    test_loss, quat_error = test_one_epoch(device, model, test_loader, args.training.loss)
    textio.cprint('Validation Loss: %f & Validation Accuracy: %s & Quat Error: %f' % (test_loss, '-', quat_error))


def train_one_epoch(device, model, train_loader, optimizer, loss):
    model.train()
    train_loss = 0.0
    pred = 0.0
    count = 0
    for i, data in enumerate(tqdm(train_loader)):
        template, source, igt = data

        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)

        output = model(template, source)

        if loss == 'cd':
            loss_val = ChamferDistanceLoss()(template, output['transformed_source'])
        elif loss == 'emd':
            loss_val = earth_mover_distance(template, output['transformed_source'])
        # print(loss_val.item())

        # forward + backward + optimize
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()
        count += 1

    train_loss = float(train_loss) / count
    return train_loss


def train(args, model, train_loader, test_loader, boardio, textio, checkpoint, device):
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.training.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.1)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_test_loss = np.inf

    for epoch in range(args.training.start_epoch, args.training.epochs):
        train_loss = train_one_epoch(device, model, train_loader, optimizer, args.training.loss)
        test_loss, quat_error = test_one_epoch(device, model, test_loader, args.training.loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            snap = {'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'min_loss': best_test_loss,
                    'optimizer': optimizer.state_dict(), }
            torch.save(snap, 'checkpoints/models/best_model_snap.t7')
            torch.save(model.state_dict(), 'checkpoints/models/best_model.t7')
            torch.save(model.feature_model.state_dict(), 'checkpoints/models/best_ptnet_model.t7')

        torch.save(snap, 'checkpoints/models/model_snap.t7')
        torch.save(model.state_dict(), 'checkpoints/models/model.t7')
        torch.save(model.feature_model.state_dict(), 'checkpoints/models/ptnet_model.t7')

        boardio.add_scalar('Train Loss', train_loss, epoch + 1)
        boardio.add_scalar('Test Loss', test_loss, epoch + 1)
        boardio.add_scalar('Best Test Loss', best_test_loss, epoch + 1)
        boardio.add_scalar('quat_error', quat_error, epoch + 1)

        textio.cprint('EPOCH:: %d, Traininig Loss: %f, Testing Loss: %f, Best Loss: %f' % (epoch + 1, train_loss, test_loss, best_test_loss))


@hydra.main(config_path='config', config_name='default')
def main(args: DictConfig):

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.training.seed)
    torch.cuda.manual_seed_all(args.training.seed)
    np.random.seed(args.training.seed)

    boardio = SummaryWriter(log_dir='checkpoints/')
    _init_(args)

    textio = IOStream('checkpoints/run.log')
    textio.cprint(OmegaConf.to_yaml(args))

    if args.data.dataset_type == 'modelnet':
        trainset = RegistrationData('PCRNet',
                                    ModelNet40Data(train=True, download=False, dir_path=to_absolute_path('data')),
                                    angle_range=args.data.registration_data.angle_range,
                                    translation_range=args.data.registration_data.translation_range)
        testset = RegistrationData('PCRNet',
                                   ModelNet40Data(train=False, download=False, dir_path=to_absolute_path('data')),
                                   angle_range=args.data.registration_data.angle_range,
                                   translation_range=args.data.registration_data.translation_range)
    elif args.data.dataset_type == 'dude':
        trainset = RegistrationData('PCRNet', DudEData(train=True))
        testset = RegistrationData('PCRNet', DudEData(train=False))
    elif args.data.dataset_type == 'dude_source':
        trainset = RegistrationData('PCRNet', DudESourceData(train=True, do_transform=True))
        testset = RegistrationData('PCRNet', DudESourceData(train=False, do_transform=True))

    train_loader = DataLoader(trainset, batch_size=args.training.batch_size, shuffle=True, drop_last=True, num_workers=args.training.workers)
    test_loader = DataLoader(testset, batch_size=args.training.batch_size, shuffle=False, drop_last=False, num_workers=args.training.workers)

    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = torch.device(f'cuda:{args.training.cuda}')

    # Create PointNet Model.
    ptnet = PointNet(emb_dims=args.pointnet.emb_dims)
    model = iPCRNet(feature_model=ptnet)
    model = model.to(device)

    checkpoint = None
    if args.training.resume:
        assert os.path.isfile(args.training.resume)
        checkpoint = torch.load(args.training.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

    if args.training.pretrained:
        pretrained = to_absolute_path(args.training.pretrained)
        assert os.path.isfile(pretrained)
        model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    model.to(device)

    if args.eval:
        test(args, model, test_loader, textio, device)
    else:
        train(args, model, train_loader, test_loader, boardio, textio, checkpoint, device)


if __name__ == '__main__':
    main()
