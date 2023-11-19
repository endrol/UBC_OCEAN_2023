import torch
import numpy as np
import os
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from utils import get_logger, set_seed, balance_accuracy_score
from data_utils.dataset import UBCOCEANDataset, get_dataloader
from model_utils.model_choose import choose_model
from tqdm import tqdm
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="parse args for training")
    parser.add_argument("--root", type=str, default="./data/", help="place to find data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=12)
    parser.add_argument("--numworker", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="bsline")
    parser.add_argument("--backbone", type=str, default="densenet169")
    parser.add_argument("--num_class", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="./output/")
    args = parser.parse_args()
    return args

def training(epoch, model, train_data, optimizer, criterion, device):
    model.train()
    losses = 0
    for imgs, labels in train_data:
        optimizer.zero_grad()
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels.to(device))
        losses += loss.item()
        loss.backward()
        optimizer.step()
    wandb.log({"train_loss": losses/len(train_data), "epoch": epoch})

def validation(epoch, model, val_data, criterion, device):
    model.eval()
    losses = 0
    y_true = []
    y_pred = []
    for imgs, labels in val_data:
        imgs = imgs.to(device)
        outputs = model(imgs)
        y_true.append(labels.item())
        y_pred.append(model.get_pred(outputs).cpu().detach().item())
        val_loss = criterion(outputs, labels.to(device))
        losses += val_loss.item()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    wandb.log({"val_loss": losses/len(val_data), "epoch": epoch})
    wandb.log({"balance_accuracy_score": balance_accuracy_score(y_true, y_pred), "epoch": epoch})
    return losses/len(val_data)

def save_ckp(ckp, save_dir):
    torch.save(ckp, save_dir)

def main():
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger('train')
    wandb.init(project="kaggle_UBCOCEAN", entity="study-sync")
    info = pd.read_csv(Path(args.root, "train.csv"))
    skf = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = choose_model(args.model_name, args)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    wandb.log({"model_info": args.model_name+args.backbone})

    min_val_loss = float("inf")
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(info, info["label"])):
        logger.info(f"fold {fold_i} training")
        train_data = UBCOCEANDataset(root=args.root, infocsv=info.iloc[train_idx])
        train_loader = get_dataloader(train_data, is_train=True, batchsize=args.batchsize, num_worker=args.numworker)
        val_data = UBCOCEANDataset(root=args.root, infocsv=info.iloc[val_idx], type="val")
        val_loader = get_dataloader(val_data, is_train=False, batchsize=1, num_worker=args.numworker)

        for epoch in tqdm(range(args.epochs)):
            training(epoch, model, train_loader, optimizer, criterion, device)
            val_loss = validation(epoch, model, val_loader, criterion, device)
            scheduler.step(val_loss)
            if val_loss < min_val_loss:
                ckp = {"model_ckp": model.state_dict()}
                save_ckp(ckp, save_dir=Path(args.save_dir, f"{args.model_name}_{args.backbone}_{args.epochs}_best.pth"))

if __name__ == "__main__":
    main()
