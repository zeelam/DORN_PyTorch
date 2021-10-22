import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from torch.optim import SGD
from tensorboardX import SummaryWriter
from models.dorn import Dorn
from modules.OrdinalRegressionLoss import OrdinalRegressionLoss
from modules.data import KittiDataset
from modules.lr_decay import PolynomialLRDecay
from torch.utils.data import DataLoader
from modules.metrics import Result, AverageMeter
from modules.utils import ImageBuilder
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def add_log(logger, avg, epoch, stage):
    logger.add_scalar(stage + '/SILog', avg.silog, epoch)
    logger.add_scalar(stage + '/RMSE', avg.rmse, epoch)
    logger.add_scalar(stage + '/absRel', avg.absrel, epoch)
    logger.add_scalar(stage + '/sqRel', avg.sqrel, epoch)
    logger.add_scalar(stage + '/Log10', avg.lg10, epoch)
    logger.add_scalar(stage + '/RMSE_log', avg.rmse_log, epoch)
    logger.add_scalar(stage + '/Delta1', avg.delta1, epoch)
    logger.add_scalar(stage + '/Delta2', avg.delta2, epoch)
    logger.add_scalar(stage + '/Delta3', avg.delta3, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments of training DORN')
    parser.add_argument('--backbone-path', type=str,
                        default='modules/backbones/pretrained_models/resnet101-imagenet.pth',
                        help='backbone weights path')
    parser.add_argument('--data-path', type=str, default='../data/kitti', help='data path')
    parser.add_argument('--training-data-path', type=str, default='data', help='training data path')
    parser.add_argument('--annotation-data-path', type=str, default='depth_annotation',
                        help='annotation data path')
    parser.add_argument('--training-file', type=str, default='eigen_train_files_with_gt.txt',
                        help='training file name')
    parser.add_argument('--validating-file', type=str, default='eigen_val_files_with_gt.txt',
                        help='validating file name')
    parser.add_argument('--output-path', type=str, default='output/dorn', help='output path')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--depth-range', nargs='+', type=int, default=[0, 80], help='[alpha, beta] depth range')
    parser.add_argument('--n-classes', type=int, default=80, help='number of intervals')
    parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--checkpoint-freq', type=int, default=1, help='frequency of checkpoint')
    parser.add_argument('--no-img-log', action='store_true', help='only output metrics in log file')
    parser.add_argument("--local_rank", default=-1, type=int)
    opt = parser.parse_args()

    local_rank = opt.local_rank
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)

    output_image = True
    if opt.no_img_log:
        output_image = False
    n_log_images = 3

    alpha = opt.depth_range[0]
    beta = opt.depth_range[1]
    K = opt.n_classes
    BATCH_SIZE = opt.batch_size
    N_EPOCH = opt.epochs
    learning_rate = opt.lr
    momentum = opt.momentum
    weight_decay = opt.weight_decay
    checkpoint_freq = opt.checkpoint_freq

    pretrained_model_path = opt.backbone_path

    model = Dorn(K=K, pretrained_model_path=pretrained_model_path)
    train_params = [{'params': model.get_1x_lr_params(), 'lr': learning_rate},
                    {'params': model.get_10x_lr_params(), 'lr': learning_rate * 10}]

    model.to(local_rank)
    if torch.cuda.device_count() > 1:
        # model = nn.DataParallel(model)
        if dist.get_rank() == 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = SGD(train_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=N_EPOCH, end_learning_rate=learning_rate * 1e-2)

    current_path = os.getcwd()
    root_path = os.path.join(current_path, opt.data_path)
    data_path = os.path.join(root_path, opt.training_data_path)
    annotation_path = os.path.join(root_path, opt.annotation_data_path)
    train_file_path = os.path.join(root_path, opt.training_file)
    val_file_path = os.path.join(root_path, opt.validating_file)

    output_path = os.path.join(current_path, opt.output_path)
    model_path = os.path.join(output_path, "models")
    log_path = os.path.join(output_path, "log")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    train_dataset = KittiDataset(data_path, annotation_path, train_file_path)
    val_dataset = KittiDataset(data_path, annotation_path, val_file_path)
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                                   sampler=torch.utils.data.distributed.DistributedSampler(train_dataset))
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                                 sampler=torch.utils.data.distributed.DistributedSampler(val_dataset))

    criterion = OrdinalRegressionLoss(alpha, beta, K)

    logger = SummaryWriter(logdir=log_path)
    last_sqrel = np.inf
    for n in range(N_EPOCH):
        if dist.get_rank() == 0:
            print('train in progress...')
        model.train()
        average_meter = AverageMeter()
        image_builder = ImageBuilder()
        train_data_loader.sampler.set_epoch(n)
        train_loader = tqdm(train_data_loader)
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            pred_labels, pred_prob = model(images)
            # calculate loss
            loss = criterion(pred_prob, labels).to(local_rank)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_depths = criterion.labelToDepth(pred_labels)

            result = Result()
            result.evaluate(pred_depths.data, labels.data, cap=80)
            average_meter.update(result, images.size(0))

            if output_image and i <= n_log_images and dist.get_rank() == 0:
                image_builder.add_row(images[0, :, :, :], labels[0, :, :], pred_depths[0, :, :])

            if dist.get_rank() == 0:
                train_loader.set_description("Loss={:.2f} "
                                             "SILog={result.silog:.2f} "
                                             "RMSE={result.rmse:.2f} "
                                             "AbsRel={result.absrel:.2f} "
                                             "SqRel={result.sqrel:.2f} "
                                             "Log10={result.lg10:.3f} "
                                             "RMSE_log={result.rmse_log:.3f} "
                                             "Delta1={result.delta1:.3f} "
                                             "Delta2={result.delta2:.3f} "
                                             "Delta3={result.delta3:.3f} "
                                             .format(loss.detach(), result=result))

        avg = average_meter.average()
        if dist.get_rank() == 0:
            add_log(logger, avg, n, "Train")
            if output_image:
                logger.add_image('Train/Image', image_builder.get_image(), n)

            print('\nEpoch: [{0}/{1}]\t'
                  'Avg_SILog={average.silog:.2f} '
                  'Avg_RMSE={average.rmse:.2f} '
                  'Avg_AbsRel={average.absrel:.2f} '
                  'Avg_SqRel={average.sqrel:.2f} '
                  'Avg_Log10={average.lg10:.3f} '
                  'Avg_RMSE_log={average.rmse_log:.3f} '
                  'Avg_Delta1={average.delta1:.3f} '
                  'Avg_Delta2={average.delta2:.3f} '
                  'Avg_Delta3={average.delta3:.3f}'.format(n + 1, N_EPOCH, average=avg))

        lr_decay.step()
        if dist.get_rank() == 0:
            print('val in progress...')
        model.eval()
        average_meter = AverageMeter()
        image_builder = ImageBuilder()
        val_data_loader.sampler.set_epoch(n)
        val_loader = tqdm(val_data_loader)
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                pred_labels, _ = model(images)
            pred_depths = criterion.labelToDepth(pred_labels)

            result = Result()
            result.evaluate(pred_depths.data, labels.data, cap=80)
            average_meter.update(result, images.size(0))

            if output_image and i <= n_log_images and dist.get_rank() == 0:
                image_builder.add_row(images[0, :, :, :], labels[0, :, :], pred_depths[0, :, :])

            if dist.get_rank() == 0:
                train_loader.set_description("SILog={result.silog:.2f} "
                                             "RMSE={result.rmse:.2f} "
                                             "AbsRel={result.absrel:.2f} "
                                             "SqRel={result.sqrel:.2f} "
                                             "Log10={result.lg10:.3f} "
                                             "RMSE_log={result.rmse_log:.3f} "
                                             "Delta1={result.delta1:.3f} "
                                             "Delta2={result.delta2:.3f} "
                                             "Delta3={result.delta3:.3f} "
                                             .format(result=result))

        avg = average_meter.average()
        if dist.get_rank() == 0:
            add_log(logger, avg, n, "Val")
            if output_image:
                logger.add_image('Val/Image', image_builder.get_image(), n)

            print('\nEpoch: [{0}/{1}]\t'
                  'Avg_SILog={average.silog:.2f} '
                  'Avg_RMSE={average.rmse:.2f} '
                  'Avg_AbsRel={average.absrel:.2f} '
                  'Avg_SqRel={average.sqrel:.2f} '
                  'Avg_Log10={average.lg10:.3f} '
                  'Avg_RMSE_log={average.rmse_log:.3f} '
                  'Avg_Delta1={average.delta1:.3f} '
                  'Avg_Delta2={average.delta2:.3f} '
                  'Avg_Delta3={average.delta3:.3f}'.format(n + 1, N_EPOCH, average=avg))

        if n % checkpoint_freq == 0 and dist.get_rank() == 0:
            print("Saving model...")
            if avg.sqrel < last_sqrel:
                print("Update best model, epoch: {}".format(n + 1))
                torch.save(model.state_dict(), os.path.join(model_path, "best.pth"))
                last_sqrel = avg.sqrel
            torch.save(model.state_dict(), os.path.join(model_path, "checkpoint_{}.pth".format(n + 1)))

    logger.close()
