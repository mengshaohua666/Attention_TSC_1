import os
import random

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datagen import CasiaDataset
from models.resnet import resnet18
from tools.train_utils import parse_args, train


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("Initializing Networks")
    model = resnet18(pretrained=True)
    model.cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().cuda()

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

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)



if __name__ == '__main__':
    main()
