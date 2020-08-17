import os
import random

import torch
from torch import nn
from torch import optim, hub
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from config import Config
from datagen import CasiaDataset
from models.resnet import resnet18
from tools.train_utils import parse_args, train, validate

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])


def main():
    args = parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("Initializing Networks")
    model = resnet18(pretrained=True)
    model.cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().cuda()

    print("Initializing Data Loader")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((0, 20)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = CasiaDataset(data_root=args.data_dir, mode='train', transform=transform)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size)
    # data_batch = next(iter(train_loader))
    # imshow(data_batch)

    val_data = CasiaDataset(data_root=args.data_dir, mode='validation', transform=transform)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size)

    print("Start Training")
    best_acc1 = 0.
    for epoch in range(1, args.epochs + 1):
        train(train_loader, model, optimizer, criterion, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)

        best_acc1 = max(acc1, best_acc1)

        if epoch == args.epochs:
            print('Save model')
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, os.path.join('weights', '{}_{}.pt'.format(args.arch, args.prefix)))


if __name__ == '__main__':
    """
    python train.py --data-dir /home/shaohua/data/Datasets/Face_Anti_Spoofing/mini-casia --arch resnet18 --epoch 10 --batch-size 16 
    """
    main()
