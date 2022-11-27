import torch
import torch.nn as nn
from torchsummary import summary
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import random
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
# from utils.provider import *
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import math
import cv2
import numpy as np

def compute_loss(loss_1, loss_2, config, niter, train_writer):
    '''
    compute the final loss for optimization
    For dVAE: loss_1 : reconstruction loss, loss_2 : kld loss
    '''
    start = config.kldweight.start
    target = config.kldweight.target
    ntime = config.kldweight.ntime

    _niter = niter - 10000
    if _niter > ntime:
        kld_weight = target
    elif _niter < 0:
        kld_weight = 0.
    else:
        kld_weight = target + (start - target) * (1. + math.cos(math.pi * float(_niter) / ntime)) / 2.

    if train_writer is not None:
        train_writer.add_scalar('Loss/Batch/KLD_Weight', kld_weight, niter)

    loss = loss_1 + kld_weight * loss_2

    return loss


def get_temp(config, niter):
    if config.get('temp') is not None:
        start = config.temp.start
        target = config.temp.target
        ntime = config.temp.ntime
        if niter > ntime:
            return target
        else:
            temp = target + (start - target) *  (1. + math.cos(math.pi * float(niter) / ntime)) / 2.
            return temp
    else:
        return 0


def rotate_data(points, rotation_dataset):
    # rotation =  rotation_dataset.sam  # B 3 3
    B = points.shape[0]
    index = [i for i in range(rotation_dataset.shape[0])]
    random.shuffle(index)
    angle = rotation_dataset[index[0:B], :]
    rotation = torch.ones((B, 3, 3))
    for j in range(B):
        r_angles = angle[j]
        Rx = torch.Tensor([[1, 0, 0],
                       [0, torch.cos(r_angles[0]), -torch.sin(r_angles[0])],
                       [0, torch.sin(r_angles[0]), torch.cos(r_angles[0])]])
        Ry = torch.Tensor([[torch.cos(r_angles[1]), 0, torch.sin(r_angles[1])],
                       [0, 1, 0],
                       [-torch.sin(r_angles[1]), 0, torch.cos(r_angles[1])]])
        Rz = torch.Tensor([[torch.cos(r_angles[2]), -torch.sin(r_angles[2]), 0],
                       [torch.sin(r_angles[2]), torch.cos(r_angles[2]), 0],
                       [0, 0, 1]])
        rotation[j] = torch.mm(Rz, torch.mm(Ry, Rx))
    rotation = rotation.cuda()
    rotated_points = torch.einsum('bij,bjk->bik', points, rotation) # points B N 3
    return rotated_points


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                              builder.dataset_builder(args, config.dataset.val)

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    summary(base_model, (1024, 3))
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        gnorm = AverageMeter()
        losses = AverageMeter(['Loss1', 'Loss2'])
        losses_rec_detail = AverageMeter(['Loss-b-coarse-pos', 'Loss-b-fine-neg', 'Loss-t-coarse-pos', 'Loss-t-fine-pos'])
        losses_klv_detail = AverageMeter(['Loss-b-klv', 'Loss-t-klv'])
        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        # for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
        for idx, (data, target) in enumerate(train_dataloader):
            data = data.float().cuda()
            # data[:, :, 0:3], angles1 = random_rotate_eco(data[:, :, 0:3], randRotationLoader)
            num_iter += 1
            n_itr = epoch * n_batches + idx
             
            data_time.update(time.time() - batch_start_time)

            temp = get_temp(config, n_itr)

            ret = base_model(data, temperature=temp, hard=False)

            loss_rec, loss_klv = base_model.module.get_loss(ret)
            loss_coarse_b_pos, loss_fine_b_neg, loss_coarse_t_pos, loss_fine_t_pos = loss_rec
            loss_klv_b, loss_klv_t = loss_klv
            loss_1 = loss_coarse_b_pos+loss_fine_b_neg+loss_coarse_t_pos+loss_fine_t_pos
            loss_2 = (loss_klv_b+loss_klv_t) * 100

            _loss = compute_loss(loss_1, loss_2, config, n_itr, train_writer)

            _loss.backward()

            if config.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_clip)
            else:
                grad_norm = misc.get_grad_norm(base_model.parameters())
            gnorm.update(grad_norm)
            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss_1 = dist_utils.reduce_tensor(loss_1, args)
                loss_2 = dist_utils.reduce_tensor(loss_2, args)
                losses.update([loss_1.item() * 1000, loss_2.item()])
            else:
                # loss_coarse_b_pos, loss_fine_b_neg, loss_coarse_t_pos, loss_fine_t_pos
                losses.update([loss_1.item() * 1000, loss_2.item()])
                losses_rec_detail.update([loss.item() * 1000 for loss in loss_rec])
                losses_klv_detail.update([loss.item() * 1000 for loss in loss_klv])

            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss_1', loss_1.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_2', loss_2.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_coarse_b_pos', loss_rec[0].item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_fine_b_neg', loss_rec[1].item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_coarse_t_pos', loss_rec[2].item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_fine_t_pos', loss_rec[3].item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_klv_b', loss_klv[0].item()* 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_klv_t', loss_klv[1].item()* 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Temperature', temp, n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f '
                          'RecLosses = %s KlvLosses = %s  GradNorm = %.4f' % (epoch, config.max_epoch, idx + 1,
                                                                              n_batches, batch_time.val(),
                                                                              data_time.val(),
                                                                              ['%.4f' % l for l in losses.val()],
                                                                              optimizer.param_groups[0]['lr'],
                                                                              ['%.4f' % l for l in losses_rec_detail.val()],
                                                                              ['%.4f' % l for l in losses_klv_detail.val()],
                                                                              gnorm.val()), logger=logger)

        if config.scheduler.type != 'function':
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_2', losses.avg(1), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s RecLosses = %s KlvLosses = %s  GradNorm = %.4f'
                  % (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                     ['%.4f' % l for l in losses_rec_detail.avg()], ['%.4f' % l for l in losses_klv_detail.avg()],
                     gnorm.avg()), logger=logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            #if metrics.better_than(best_metrics):
            #    best_metrics = metrics
            #    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        #builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        #if (config.max_epoch - epoch) < 5:
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)   
    if train_writer is not None:  
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLoss1', 'SparseLoss2', 'DenseLoss1', 'DenseLoss2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (data, taxonomy_id) in enumerate(test_dataloader):
            taxonomy_id = str(taxonomy_id.item())
            points = data.float().cuda()

            ret = base_model(inp = points, hard=True, eval=True)
            coarse_points_1 = ret[0]
            dense_points_1 = ret[1]
            coarse_points_2 = ret[2]
            dense_points_2 = ret[3]
            # trans_logit = ret[-1]# B N == 1 4096
            #label = ret[-1].data.max(1)[1]

            sparse_loss1 = ChamferDisL1(coarse_points_1, points)
            dense_loss1 = ChamferDisL1(dense_points_1, points)

            sparse_loss2 = ChamferDisL1(coarse_points_2, points)
            dense_loss2 = ChamferDisL1(dense_points_2, points)


            # if args.distributed:
            #     sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
            #     sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
            #     dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
            #     dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss1.item() * 1000, sparse_loss2.item() * 1000, dense_loss1.item() * 1000, dense_loss2.item() * 1000])

            _metrics = Metrics.get(dense_points_2, points)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            vis_list = [0, 500, 1000, 1500, 2000]
            if val_writer is not None and idx in vis_list:
                original_pc = data.squeeze().detach().cpu().numpy()
                original_pc = misc.get_ptcloud_img(original_pc, -1)
                val_writer.add_image('Model%02d/Original' % idx, original_pc, epoch, dataformats='HWC')
                
                sparse = coarse_points_1.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse, -1)
                val_writer.add_image('Model%02d/Sparse1' % idx, sparse_img, epoch, dataformats='HWC')

                dense = dense_points_1.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense, -1)
                val_writer.add_image('Model%02d/Dense1' % idx, dense_img, epoch, dataformats='HWC')

                sparse = coarse_points_2.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse, -1)
                val_writer.add_image('Model%02d/Sparse2' % idx, sparse_img, epoch, dataformats='HWC')

                dense = dense_points_2.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense, -1)
                val_writer.add_image('Model%02d/Dense2' % idx, dense_img, epoch, dataformats='HWC')

            #vis_list = list(range(100))
            #if epoch>=298 and idx in vis_list:
            #if idx in vis_list:
            #    original_pc = data.squeeze().detach().cpu().numpy()
            #    original_pc = misc.get_ptcloud_img(original_pc, -1)
            #   val_writer.add_image('label%d/Original' % label, original_pc, epoch, dataformats='HWC')

            if (idx+1) % 2000 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)

        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    #shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    modelnet_dict = {0:'airplane',1:'bathtub',2:'bed',3:'bench',4:'bookshelf',5:'bottle',6:'bowl',7:'car',8:'chair',9:'cone',10:'cup',11:'curtain',12:'desk',13:'door',14:'dresser',15:'flower_pot',16:'glass_box',17:'guitar',18:'keyboard',19:'lamp',20:'laptop',21:'mantel',22:'monitor',23:'night_stand',24:'person',25:'piano',26:'plant',27:'radio',28:'range_hood',29:'sink',30:'sofa',31:'stairs',32:'stool',33:'table',34:'tent',35:'toilet',36:'tv_stand',37:'vase',38:'wardrobe',39:'xbox'}
    print_log('============================ TEST RESULTS ============================', logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += modelnet_dict[int(float(taxonomy_id))] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse1', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense1', test_losses.avg(1), epoch)
        val_writer.add_scalar('Loss/Epoch/Sparse2', test_losses.avg(2), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense2', test_losses.avg(3), epoch)

        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)

def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    target = './vis'
    useful_cate = [
        "02691156",
        "02818832",
        "04379243",
        "04099429",
        "03948459",
        "03790512",
        "03642806",
        "03467517",
        "03261776",
        "03001627",
        "02958343",
        "03759954"
    ]
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            if  taxonomy_ids[0] not in useful_cate:
                continue
    
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')


            ret = base_model(inp = points, hard=True, eval=True)
            dense_points = ret[1]

            final_image = []

            data_path = f'./vis/{taxonomy_ids[0]}_{idx}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            points = points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'gt.txt'), points, delimiter=';')
            points = misc.get_ptcloud_img(points)
            final_image.append(points)

            dense_points = dense_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'dense_points.txt'), dense_points, delimiter=';')
            dense_points = misc.get_ptcloud_img(dense_points)
            final_image.append(dense_points)

            img = np.concatenate(final_image, axis=1)
            img_path = os.path.join(data_path, f'plot.jpg')
            cv2.imwrite(img_path, img)

            if idx > 1000:
                break

        return 
