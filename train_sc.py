import pandas as pd
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from data_sc import load_df_w_features, TrainDataset, ValDataset, de_to_tscore, tscore_to_de
from model_sc import test_MLP, CustomTransformer_v3, ResMLP, ResMLP_BN, cVAE

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

batch_size = 612
epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "mps")


class CustomLoss(nn.Module):
    def __init__(self, mae_coeff, mse_coeff, rmse_coeff, huber_coeff, bce_coeff):
        super(CustomLoss, self).__init__()
        self.mae_coeff = mae_coeff
        self.mse_coeff = mse_coeff
        self.rmse_coeff = rmse_coeff
        self.huber_coeff = huber_coeff
        self.bce_coeff = bce_coeff

    def forward(self, y, y_h):
        mae_loss = self.mae_coeff*F.l1_loss(y,y_h)
        mse_loss = self.mse_coeff*F.mse_loss(y,y_h)
        rmse_loss = self.rmse_coeff*torch.sqrt(F.mse_loss(y,y_h))
        huber_loss = self.huber_coeff*F.huber_loss(y,y_h)
        bce_loss = self.bce_coeff*F.binary_cross_entropy(torch.sigmoid(y),torch.sigmoid(y_h))
        loss = mae_loss + mse_loss + rmse_loss + huber_loss + bce_loss
        return loss


class CustomLoss_no_BCE(nn.Module):
    def __init__(self, mae_coeff, mse_coeff, rmse_coeff, huber_coeff):
        super(CustomLoss_no_BCE, self).__init__()
        self.mae_coeff = mae_coeff
        self.mse_coeff = mse_coeff
        self.rmse_coeff = rmse_coeff
        self.huber_coeff = huber_coeff

    def forward(self, y, y_h):
        mae_loss = self.mae_coeff*F.l1_loss(y,y_h)
        mse_loss = self.mse_coeff*F.mse_loss(y,y_h)
        rmse_loss = self.rmse_coeff*torch.sqrt(F.mse_loss(y,y_h))
        huber_loss = self.huber_coeff*F.huber_loss(y,y_h)
        loss = mae_loss + mse_loss + rmse_loss + huber_loss
        return loss


class cVAE_Loss(nn.Module):
    def __init__(self, mae_coeff, mse_coeff, rmse_coeff, huber_coeff):
        super(cVAE_Loss, self).__init__()
        self.reconstruction_loss = nn.HuberLoss()

    def forward(self, tup, y):
        (y_h, mean, logvar) = tup
        reconstruction_loss = self.reconstruction_loss(y, y_h)
        KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
        return KL_divergence + reconstruction_loss


def leave_one_out_cv(df, model, optimizer, lr_scheduler, criterion, config, weights=False):
    b_cells = df.loc[df["cell_type"] == "B cells"]
    myeloid_cells = df.loc[df["cell_type"] == "Myeloid cells"]
    other_cells = df.loc[
        (df["cell_type"] != "Myeloid cells") & (df["cell_type"] != "B cells")
    ]

    score = 0
    weights = []
    lr_init = lr_scheduler.state_dict()
    optim_init = optimizer.state_dict()
    model_init = model.state_dict()

    for i in range(len(b_cells)):
        val_df = pd.concat([b_cells.iloc[[i]], myeloid_cells.iloc[[i]]], axis=0, ignore_index=True)
        train_df = pd.concat(
            [
                other_cells,
                b_cells.iloc[0:i],
                b_cells.iloc[i + 1 : len(b_cells)],
                myeloid_cells.iloc[0:i],
                myeloid_cells.iloc[i + 1 : len(b_cells)],
            ],
            ignore_index=True,
        )
        train_dataset = TrainDataset(train_df, config, device)
        val_dataset = ValDataset(val_df, config, train_dataset)

        lr_scheduler.load_state_dict(lr_init)
        optimizer.load_state_dict(optim_init)
        model.load_state_dict(model_init)
        model = model.to(device)

        best_score, best_weights = train_best_val_score(train_dataset, val_dataset, model, optimizer, lr_scheduler, criterion, config)
        weights.append(best_weights)
        score += best_score
    score /= len(b_cells)
    if weights:
        return score, weights
    else:
        return score


def mrrmse(y, y_h):
    return (((y - y_h)**2).mean(dim=1)**0.5).mean()


def train_best_val_score(train_dataset, val_dataset, model, optimizer, lr_scheduler, criterion, config):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.apply(init_weights)

    best_past_val_loss = float("inf")
    best_weights = model.state_dict()

    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for batch, x in enumerate(train_dataloader):
            optimizer.zero_grad()
            out = model(x[0:(len(x)-1)])
            y = x[-1]
            loss = criterion(out, y)
            loss.backward()
            if config["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for batch, x in enumerate(val_dataloader):
                out = model(x[0:(len(x)-1)])
                if config["inv_transform"]:
                    if config["svd"]:
                        out = torch.as_tensor(train_dataset.svd.inverse_transform(out.detach().cpu()), dtype=torch.float, device=device)
                    elif config["pca"]:
                        out = torch.as_tensor(train_dataset.pca.inverse_transform(out.detach().cpu()), dtype=torch.float, device=device)
                    else:
                        raise Exception("No inverse transform without pca or svd")
                if config["t_score"]:
                    out = torch.as_tensor(tscore_to_de(out.detach().cpu().numpy()), dtype=torch.float, device=device)
                loss = mrrmse(out, x[-1])
                val_loss += loss.item()
            val_loss /= len(val_dataloader)

            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

            if val_loss < best_past_val_loss:
                best_past_val_loss = val_loss
                best_weights = model.state_dict()

            lr_scheduler.step()
    print(f"Best Val Loss: {best_past_val_loss}")
    return best_past_val_loss, best_weights


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def complete_cv_loop():
    config = {
        "emb_cell": True,
        "emb_sm": True,
        "mean_cell": False,
        "mean_sm": False,
        "std_cell": True,
        "std_sm": True,
        "median_cell": False,
        "median_sm": False,
        "control": False,
        "t_score": True,
        "svd": True,
        "pca": False,
        "inv_transform": True,
        "chemberta":True,
        "chemberta_mean":False,
        "noise": True,
        "grad_clip": True
    }
    df = load_df_w_features()
    model = test_MLP(config)
    optimizer = torch.optim.AdamW(model.parameters(), 2e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)
    criterion = CustomLoss(0.5158699762912065, 0.17759311956146973, 0.48735628809909753, 0.4192700658080418, 0.510095994098853)
    score = leave_one_out_cv(df, model, optimizer, lr_scheduler, criterion, config)
    print(score)


if __name__ == "__main__":
    complete_cv_loop()