from dataset import load_dataset
import easydict
import json
import wandb
import os
import random
from tqdm import tqdm

from importlib import import_module


import pandas as pd
from PIL import Image
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms, utils
from loss import create_criterion

from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
from sklearn.metrics import f1_score
# 현재 OS 및 라이브러리 버전 체크 체크
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    """
    동일한 조건으로 학습을 할 때, 동일한 결과를 얻기 위해 seed를 고정시킵니다.

    Args:
        seed: seed 정수값
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


seed_everything(1004)


def train(args):

    # model save 경로 지정하기
    save_dir = f"./results/{args.target}"
    os.makedirs(save_dir, exist_ok=True)

    # 데이터셋 로드하기
    trn_dataset = load_dataset(
        dataset=args.dataset, target=args.target, train=True)
    val_dataset = load_dataset(
        dataset=args.dataset, target=args.target, train=False)

    transform_original = getattr(import_module(
        "dataset"), args.augmentation_original)  # 원래 데이터셋에 대한 augmentation
    
    transform_aaf = getattr(import_module(
        "dataset"), args.augmentation_aaf)           # 추가 데이터셋에 대한 augmentation

    transform_crop = getattr(import_module(
        "dataset"), args.augmentation_crop)

    transform_test = getattr(import_module(
        "dataset"), args.augmentation_test)

    transform = {
        'original': transform_original(),
        'aaf': transform_aaf(),
        'crop':transform_crop(),
        'test':transform_test()
    }



    trn_dataset.set_transform(transform)
    val_dataset.set_transform(transform)

    trn_loader = DataLoader(
        trn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    num_class = len(trn_dataset.classes)
    model_module = getattr(import_module("model"), args.model)
    model = model_module(num_class=num_class).to(device)

    # Weighted Cross Entroy Loss
    weights = [1-n/sum(trn_dataset.count) for n in trn_dataset.count]
    weights = torch.FloatTensor(weights).to(device)

    criterion = create_criterion(args.criterion, weight=weights).cuda()

    # optimizer
    optimizer_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = optimizer_module(model.parameters(), lr=args.lr)

    # Scheduler
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer, lr_lambda=lambda epoch: 0.95**epoch)

    wandb.init(
        project=args.project,
        config={
            "learning_rate": args.lr,
            "architecture": args.model,
            "dataset": args.dataset,
        }
    )

    best_val_acc = 0.0
    best_val_f1 = 0.0

    epochs = args.epochs

    # Training Start!
    print("Start Training!!")
    for epoch in range(epochs):
        running_loss = 0.0

        total = 0
        correct = 0
        lr = scheduler.get_last_lr()[0]

        model.train()
        for inputs, labels in tqdm(trn_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy 계산
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        scheduler.step()
        acc = correct/total
        print(
            f"[TRN]EPOCH:{epoch+1}, LR:{lr}, loss:{running_loss/len(trn_loader):.7f}, acc:{100*acc:.2f}%")

        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            f1 = 0.0
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                # Accuracy 계산
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

                f1 += f1_score(preds.cpu().numpy(),
                               labels.cpu().numpy(), average='macro')
            val_acc = correct/total
            print(
                f"[VAL]EPOCH:{epoch+1}, f1:{f1/len(val_loader):.3f}, val_acc:{100*val_acc:.2f}%")

            f1 = f1/len(val_loader)

            # 모델 저장
            if f1 > best_val_f1:
                print("New Best Model for F1 Score! saving the model...")
                torch.save(model.state_dict(
                ), f"{save_dir}/{args.model}_epoch{epoch:03}_f1_{f1:4.2%}.ckpt")
                best_val_f1 = f1

            # print("saving the Every model...")
            # torch.save(model.state_dict(
            # ), f"{save_dir}/{args.model}_epoch{epoch:03}_f1_{f1:4.2%}.ckpt")
            # if f1 > best_val_f1:
            #     best_val_f1 = f1

            if val_acc > best_val_acc:
                if f1 == best_val_f1:
                    continue
                print("New Best Model for Acc Score! saving the model...")
                torch.save(model.state_dict(
                ), f"{save_dir}/{args.model}_epoch{epoch:03}_acc_{val_acc:4.2%}.ckpt")
                best_val_acc = val_acc

        wandb.log({"acc": acc, "loss": running_loss /
                  len(trn_loader), "val_acc": val_acc, "f1": f1})
    wandb.finish()


if __name__ == '__main__':
    with open('./args.json', 'r') as f:
        args = easydict.EasyDict(json.load(f))
    train(args)
