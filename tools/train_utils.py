import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from tools.model_utils import AverageMeter, ProgressMeter, accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="", required=True)
    parser.add_argument("--arch", type=str, default="resnet18", help="backbone architecture")
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    return args


def imshow(data):
    """Imshow for Tensor."""
    rgb_tensors = data['rgb']
    msr_tensors = data['msr']
    fig = plt.figure(figsize=(8, 8))
    for index, sample in enumerate(zip(rgb_tensors, msr_tensors)):
        rgb_tensor, msr_tensor = sample
        # rgb
        inp = rgb_tensor.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        fig.add_subplot(4, 4, 2 * index + 1)
        plt.imshow(inp)
        # msr
        fig.add_subplot(4, 4, 2 * index + 2)
        plt.imshow(msr_tensor[:, :, 0], cmap='gray')
    plt.show()


def train(train_loader, model, optimizer, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses, top1], prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for batch_idx, (images, target) in enumerate(train_loader):
        images, target = images.cuda(), target.cuda()
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, top1], prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images, target = images.cuda(), target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target)
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return top1.avg
