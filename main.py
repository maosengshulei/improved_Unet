import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess
import sys
import pytz
import torch
import yaml
sys.path.append('E:\\VOC_data\\TernausNet')
import unet_models
import plaque
import unet_trainer
from resnet_unet import ModelBuilder, SegmentationModule

configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=100000,
        #lr=1.0e-10,
        #momentum=0.99,
        #weight_decay=0.0005,
        interval_validate=400,
    )
}


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash


def get_log_dir(model_name, config_id, cfg):
    # load config
    name = 'MODEL-%s_CFG-%03d' % (model_name, config_id)
    for k, v in cfg.items():
        v = str(v)
        if '/' in v:
            continue
        name += '_%s-%s' % (k.upper(), v)
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    name += '_VCS-%s' % git_hash()
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    # create out
    log_dir = osp.join(here, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN32s,
        torchfcn.models.FCN16s,
        torchfcn.models.FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))


here = osp.dirname(osp.abspath(__file__))

def create_optimizers(nets, args):
    (net_encoder, net_decoder) = nets
    optimizer_encoder = torch.optim.SGD(
        net_encoder.parameters(),
        lr=args.lr_encoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        net_decoder.parameters(),
        lr=args.lr_decoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    return (optimizer_encoder, optimizer_decoder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('--use_resnet', type=int,default=True)
    parser.add_argument('--arch_encoder', default='resnet50',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='c2_bilinear',
                        help="architecture of net_decoder")
    parser.add_argument('--weights_encoder', default='',
                        help="weights to finetune net_encoder")
    parser.add_argument('--weights_decoder', default='',
                        help="weights to finetune net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')
    parser.add_argument('--lr_encoder', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_decoder', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--fix_bn', default=0, type=int,
                        help='fix bn params')

    parser.add_argument('--resume', help='Checkpoint path')

    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    out = get_log_dir('unet11', args.config, cfg)
    resume = args.resume
    use_resnet=args.use_resnet

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    #train_loader = torch.utils.data.DataLoader(
     #   torchfcn.datasets.SBDClassSeg(root, split='train', transform=True),
      #  batch_size=1, shuffle=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(
        plaque.Plaqueseg(root, split='train', transform=True),
        batch_size=2, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        plaque.Plaqueseg(
            root, split='val', transform=True),
        batch_size=2, shuffle=False, **kwargs)

    # 2. model


    if use_resnet:
        builder = ModelBuilder()
        net_encoder = builder.build_encoder(
            arch=args.arch_encoder,
            fc_dim=args.fc_dim,
            weights=args.weights_encoder)
        net_decoder = builder.build_decoder(
            arch=args.arch_decoder,
            fc_dim=args.fc_dim,
            num_class=1,
            weights=args.weights_decoder)
        model = SegmentationModule(
            net_encoder, net_decoder)
        nets=(net_encoder,net_decoder)
    else:
        model = unet_models.unet11(pretrained=False)



    start_epoch = 0
    start_iteration = 0
    if resume and use_resnet:
        checkpoint = torch.load(resume)
        net_encoder.load_state_dict(checkpoint['encoder_model_state_dict'])
        net_decoder.load_state_dict(checkpoint['decoder_model_state_dict'])
        model=SegmentationModule(net_encoder,net_decoder)
        nets=(net_encoder,net_decoder)
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    if resume and not use_resnet:
        checkpoint=torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']

    if cuda:
        model = model.cuda()

    # 3. optimizer

    #optim = lambda lr: torch.optim.Adam(model.parameters(), lr=lr)
    if use_resnet:
        optim = create_optimizers(nets, args)
    else:
        optim=torch.optim.SGD(
            model.parameters(),
            lr=2e-2,
            momentum=0.9,
            weight_decay=1e-4)


    trainer = unet_trainer.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        use_resnet=use_resnet,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
