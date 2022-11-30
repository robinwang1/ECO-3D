from tensorboardX import SummaryWriter
import os
import sys
import torch
import numpy as np
import math
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import datetime
import logging
import provider
import importlib
import shutil
import argparse
from utils.RobustPointSet import *
from utils.ScanObjectNNDataset import *
from utils.AverageMeter import AverageMeter
from utils import misc
import time
from autoencoder.dvae import *
import torch.nn.functional as F
# from models.dgcnn_cls import *
# from torch.utils.data import TensorDataset, DataLoader
from provider import ECOLoss

from pathlib import Path
from tqdm import tqdm

class balance_weight(torch.nn.Module):
    def __init__(self):
        super(balance_weight, self).__init__()
        self.weight = torch.nn.parameter(torch.tensor(0.5))
    def forward(self, loss_con, loss_equ):
        total_loss = self.weight * loss_con +  (1-self.weight) * loss_equ
        return total_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--data_root', type=str, default='data/', help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='pointnet_eco_cls', help='model name [default: pointnet_eco_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=500, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--grad_clip', type=float, default=1, help='clipping grad')
    parser.add_argument('--use_pretrained', action='store_true', default=False)
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_eco', action='store_true', default=False, help='use ecoloss')
    parser.add_argument('--trans', type=str, default='original', help='transformations')
    parser.add_argument('--dataset', type=str, default='RobustPointSet', help='transformations')
    parser.add_argument('--use_con', action='store_true', default=False, help='use contrastive loss')
    parser.add_argument('--use_equ', action='store_true', default=False, help='use equivariant loss')

    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def cal_loss(pred, eps=0):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gt1, gt2 = pred[2], pred[3]
    pred1, pred2 = pred[0], pred[1]
    # gt1 = gt1.reshape(-1, num_token)
    # gt2 = gt2.reshape(-1, num_token)
    # gt1 = gt1.argmax(-1).long()  # B Gt
    # gt2 = gt2.argmax(-1).long()  # B Gt
    gt1 = gt1.contiguous().view(-1)
    gt1 = gt1.to(torch.int64)
    gt2 = gt2.contiguous().view(-1)
    gt2 = gt2.to(torch.int64)

    n_class = pred1.size(1)

    one_hot1 = torch.zeros_like(pred1).scatter(1, gt1.view(-1, 1), 1)
    one_hot1 = one_hot1 * (1 - eps) + (1 - one_hot1) * eps / (n_class - 1)

    one_hot2 = torch.zeros_like(pred2).scatter(1, gt2.view(-1, 1), 1)
    one_hot2 = one_hot2 * (1 - eps) + (1 - one_hot2) * eps / (n_class - 1)

    loss1 = -(one_hot1 * pred1).sum(dim=1).mean()
    loss2 = -(one_hot2 * pred2).sum(dim=1).mean()

    return (loss1 + loss2) / 2


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('eco3d')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.dataset)
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.trans)
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.model)
    exp_dir.mkdir(exist_ok=True)

    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    path = args.data_root + args.dataset + '/'
    if args.dataset == 'RobustPointSet':
        num_class = 40
        num_tokens = 9840
        train_dataset = RobustPointSet(path, args.trans, 'train')
        test_dataset = RobustPointSet(path, args.trans, 'test')

    elif args.dataset == 'ScanObjectNN':
        num_class = 15
        if args.trans == 'simple':
            num_tokens = 2312
            train_dataset = ScanObjectNN(path, 'train')
            test_dataset = ScanObjectNN(path, 'test')

        elif args.trans == 'midium':
            num_tokens = 11416
            train_dataset = ScanObjectNN_midium(path, 'train')
            test_dataset = ScanObjectNN_midium(path, 'test')

        elif args.trans == 'hard':
            num_tokens = 11416
            train_dataset = ScanObjectNN_hardest(path, 'train')
            test_dataset = ScanObjectNN_hardest(path, 'test')

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    train_writer = SummaryWriter(os.path.join(str(exp_dir), 'train'), filename_suffix=f'-{timestamp}')
    val_writer = SummaryWriter(os.path.join(str(exp_dir), 'test'), filename_suffix=f'-{timestamp}')

    '''TRAIN MODEL LOADING'''
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/%s_utils.py' % args.model, str(exp_dir))
    shutil.copy('train_eco3d_cls.py', str(exp_dir))


    vae = DiscreteVAEHDRm(num_tokens).cuda().eval()
    dvae_ckpt = 'pretrained/dvaehdrm/' + args.dataset + '/' + args.trans + '/ckpt.pth'
    ckpt = torch.load(dvae_ckpt, map_location='cpu')
    base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
    vae.load_state_dict(base_ckpt, strict=True)
    for param in vae.parameters():
        param.requires_grad = False
    shutil.copy('autoencoder/dvae.py', str(exp_dir))
    bw = balance_weight().cuda().train()
    for param in bw.parameters():
        param.requires_grad = True

    if not args.use_cpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        classifier = model.get_model(num_tokens, num_class, dvae_ckpt=dvae_ckpt)
        criterion = model.get_loss()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        ECOLoss = provider.ECOLoss(batch_size=args.batch_size, use_eco=False).cuda()
    else:
        classifier = model.get_model(num_tokens, num_class, dvae_ckpt=dvae_ckpt)
        criterion = model.get_loss()
        ECOLoss = provider.ECOLoss(batch_size=args.batch_size, use_eco=False)
    classifier.apply(inplace_relu)

    if args.use_pretrained:
        try:
            checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrain model')
        except:
            log_string('No existing model, starting training from scratch...')
            start_epoch = 0
    else:
        start_epoch = 0

    if args.model in ['pointnet_eco_cls']:
        optimizer_p = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate / 2,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate / 2
        )
        scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=20, gamma=0.7)
    elif args.model in ['dgcnn_eco_cls']:
        optimizer_p = torch.optim.SGD(classifier.parameters(), lr=0.1 / 2, momentum=0.9, weight_decay=1e-4 / 2)
        scheduler_p = CosineAnnealingLR(optimizer_p, args.epoch, eta_min=0.001)

    elif args.model in ['pct_eco_cls']:
        optimizer_p = torch.optim.SGD(classifier.parameters(), lr=0.1 / 2, momentum=0.9, weight_decay=1e-4 / 2)
        scheduler_p = CosineAnnealingLR(optimizer_p, args.epoch, eta_min=0.001)

    if args.model in ['pointnet_eco_cls']:
        optimizer_f = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
        scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_f, step_size=20, gamma=0.7)
    elif args.model in ['dgcnn_eco_cls']:
        optimizer_f = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler_f = CosineAnnealingLR(optimizer_f, args.epoch, eta_min=0.001)

    elif args.model in ['pct_eco_cls']:
        optimizer_f = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler_f = CosineAnnealingLR(optimizer_f, args.epoch, eta_min=0.001)

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    pre_train_epoch = 100

    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch, epoch, args.epoch-1))
        epoch_start_time = time.time()
        batch_start_time = time.time()
        gnorm = AverageMeter()
        losses = AverageMeter(['Loss_cls', 'Loss_con', 'Loss_equ'])
        acc = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        classifier = classifier.train()
        n_batches = len(trainDataLoader)
        for idx, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                          smoothing=0.9):
            if epoch < pre_train_epoch:
                optimizer = optimizer_p
            else:
                optimizer = optimizer_f
            optimizer.zero_grad()
            points = points.numpy()
            points = provider.random_point_dropout(points)
            points = provider.shift_point_cloud(points)
            points = torch.Tensor(points)
            points, target = points.cuda(), target.cuda()
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)

            with torch.no_grad():
                neighborhood_b, center_b = vae.group_divider_b(points.contiguous())  # B Gb Gb/2 3;  B Gb 3
                feature_b = vae.encoder_b(neighborhood_b)  # B Gb Cb
                # logits_b = dvaehdrm.dgcnn_b(feature_b, center_b)  # B Gb Nb
                # print(logits_b[:, 1, :])
                # print(torch.mean(logits_b, dim=1))
                neighborhood_t, center_t = vae.group_divider_t(feature_b.contiguous())  # B Gt Gt/2 Cb;  B Gt Cb
                feature_t = vae.encoder_t(neighborhood_t)  # B Gt Ct;
                logits_t = vae.dgcnn_t(feature_t, center_t)  # B Gt Nt

                # print(logits_t[:, 1, :])
                # print(torch.mean(logits_t, dim=1))

                # logits_t = F.log_softmax(logits_t, dim=1).view(-1, 128)
            if epoch < pre_train_epoch:
                pred_cls, equivariant, contrast = classifier(feature_b, neighborhood_t, center_t, logits_t, fine_tune=False)
                if args.use_equ:
                    loss_equ = cal_loss(equivariant)
                else:
                    loss_equ = torch.tensor(0.0).cuda()
                if args.use_con:
                    loss_con = ECOLoss(contrast[0], contrast[1], None)
                else:
                    loss_con = torch.tensor(0.0).cuda()
                if not args.use_equ and not args.use_con:
                    loss_cls = criterion(pred_cls, target.long())
                else:
                    loss_cls = torch.tensor(0.0).cuda()
                if args.use_equ and args.use_con:
                    loss = bw(loss_con, loss_equ)
                else:
                    loss = loss_cls + loss_con + loss_equ
            else:
                loss_equ = torch.tensor(0.0).cuda()
                loss_con = torch.tensor(0.0).cuda()
                pred_cls = classifier(feature_b, neighborhood_t, center_t, logits_t, fine_tune=True)
                loss_cls = criterion(pred_cls, target.long())
                loss = loss_cls + loss_con + loss_equ

            pred_choice = pred_cls.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            accuracy = correct.item() / float(points.size()[0])

            acc.update(accuracy)
            losses.update([loss_cls.item(), loss_con.item(), loss_equ.item()])
            loss.backward()

            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_clip)
            else:
                grad_norm = misc.get_grad_norm(classifier.parameters())
            gnorm.update(grad_norm)

            if epoch< pre_train_epoch:
                optimizer = optimizer_p
            else:
                optimizer = optimizer_f
            optimizer.step()
            global_step += 1

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_cls', loss_cls.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_con', loss_con.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_equ', loss_equ.item(), n_itr)
                train_writer.add_scalar('Accuracy/Batch', accuracy, n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                log_string('[Epoch %d/%d][Batch %d/%d] Losses = %s lr = %.6f '
                           'Accuracy = %.3f  GradNorm = %.4f' % (epoch, args.epoch, idx + 1,
                                                                 n_batches, ['%.4f' % l for l in losses.val()],
                                                                 optimizer.param_groups[0]['lr'],
                                                                 acc.val(),
                                                                 gnorm.val()))
        if epoch < pre_train_epoch:
            scheduler = scheduler_p
        else:
            scheduler = scheduler_f
        scheduler.step()
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_cls', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_con', losses.avg(1), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_equ', losses.avg(2), epoch)
            train_writer.add_scalar('Accuracy/Epoch', acc.avg(), epoch)

        log_string('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s Accuracy = %.3f GradNorm = %.4f'
                   % (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                      acc.avg(),
                      gnorm.avg()))

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), vae, testDataLoader, val_writer, epoch,
                                           num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


def test(model, model_d, loader, val_writer, epoch, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        # points = pc_normalize(points.numpy())
        # points = torch.Tensor(points)
        points, target = points.float().cuda(), target.long().cuda()
        with torch.no_grad():
            neighborhood_b, center_b = model_d.group_divider_b(points.contiguous())  # B Gb Gb/2 3;  B Gb 3
            feature_b = model_d.encoder_b(neighborhood_b)  # B Gb Cb
            neighborhood_t, center_t = model_d.group_divider_t(feature_b.contiguous())  # B Gt Gt/2 Cb;  B Gt Cb

        pred = classifier(feature_b, neighborhood_t, center_t, None, fine_tune=True)

        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    val_writer.add_scalar('Acc/Epoch/Instance', instance_acc, epoch)
    val_writer.add_scalar('Acc/Epoch/Classwise', class_acc, epoch)

    return instance_acc, class_acc


if __name__ == '__main__':
    args = parse_args()
    main(args)