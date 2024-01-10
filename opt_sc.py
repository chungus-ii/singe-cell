import os
import optuna
import torch
import torch.nn as nn
from data_sc import load_df_w_features, calculate_dim
from train_sc import leave_one_out_cv, CustomLoss, cVAE_Loss
from model_sc import test_MLP, CustomTransformer_v3, ResMLP, ResMLP_BN, cVAE

lr = 1e-3

def data_objective(trial):
    dim_reduction = trial.suggest_categorical("Dimensionality Reduction", ["None", "Truncated SVD w/o Inverse Transform", "Truncated SVD", "PCA w/o Inverse Transform", "PCA"])
    if dim_reduction == "None":
        pca = False
        svd = False
        inv = False
    elif dim_reduction == "Truncated SVD w/o Inverse Transform":
        pca = False
        svd = True
        inv = False
    elif dim_reduction == "Truncated SVD":
        pca = False
        svd = True
        inv = True
    elif dim_reduction == "PCA w/o Inverse Transform":
        pca = True
        svd = False
        inv = False
    elif dim_reduction == "PCA":
        pca = True
        svd = False
        inv = True
    else:
        raise Exception(f"Huh?: {dim_reduction}")
    embedding_table = trial.suggest_categorical("Embedding Tables", [True, False])
    means = trial.suggest_categorical("Cell and Compound Means per gene", [True, False])
    stds = trial.suggest_categorical("Cell and Compound Standard Deviations per gene", [True, False]),
    medians = trial.suggest_categorical("Cell and Compound Medians per gene", [True, False])
    config = {
        "emb_cell": embedding_table,
        "emb_sm": embedding_table,
        "mean_cell": means,
        "mean_sm": means,
        "std_cell": stds,
        "std_sm": stds,
        "median_cell": medians,
        "median_sm": medians,
        "control": False,
        "t_score": trial.suggest_categorical("T-score Transform", [True, False]),
        "svd": svd,
        "pca": pca,
        "inv_transform": inv,
        "chemberta": False,
        "chemberta_mean": False,
        "noise": False,
        "grad_clip": False
    }
    df = load_df_w_features()
    model = test_MLP(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)
    criterion = nn.HuberLoss()
    score = leave_one_out_cv(df, model, optimizer, lr_scheduler, criterion, config)
    return score


def data_study(BASE_PATH="optuna/data_study.sqlite3"):
    search_space = {
        "Dimensionality Reduction": ["None", "Truncated SVD w/o Inverse Transform", "Truncated SVD", "PCA w/o Inverse Transform", "PCA"],
        "Embedding Tables": [True, False],
        "Cell and Compound Means per gene": [True, False],
        "Cell and Compound Standard Deviations per gene": [True, False],
        "Cell and Compound Medians per gene": [True, False],
        "T-score Transform": [True, False]
    }
    ABSOLUTE_PATH = "sqlite:///" + os.path.join(os.path.dirname(__file__), BASE_PATH)
    rdb_raw_bytes_url = r'{}'.format(ABSOLUTE_PATH)
    study = optuna.create_study(
        storage=ABSOLUTE_PATH,
        sampler=optuna.samplers.GridSampler(search_space),
        study_name="Data Preprocessing"
    )
    study.optimize(data_objective)
    print(f"Best value: {study.best_value} (params: {study.best_params})")


def loss_objective(trial):
    mae_coeff = trial.suggest_float("MAE coefficient",0,1)
    mse_coeff = trial.suggest_float("MSE coefficient",0,1)
    rmse_coeff = trial.suggest_float("RMSE coefficient",0,1)
    huber_coeff = trial.suggest_float("Huber coefficient",0,1)
    bce_coeff = trial.suggest_float("BCE coefficient",0,1)
    
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
        "chemberta": False,
        "chemberta_mean": False,
        "noise": False,
        "grad_clip": False
    }

    df = load_df_w_features()
    model = test_MLP(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)
    criterion = CustomLoss(mae_coeff, mse_coeff, rmse_coeff, huber_coeff, bce_coeff)
    score = leave_one_out_cv(df, model, optimizer, lr_scheduler, criterion, config)
    return score


def loss_study(BASE_PATH="optuna/loss_study_cmaes.sqlite3"):
    ABSOLUTE_PATH = "sqlite:///" + os.path.join(os.path.dirname(__file__), BASE_PATH)
    study = optuna.create_study(
        storage=ABSOLUTE_PATH,
        sampler=optuna.samplers.CmaEsSampler(),
        study_name="Loss Function - CMA-ES Sampler"
    )
    study.optimize(loss_objective, n_trials=50)
    print(f"Best value: {study.best_value} (params: {study.best_params})")


def model_objective(trial):
    suggest = trial.suggest_categorical("Model", ["MLP", "MLP with Residual Connections", "MLP with Residual Connections and BN", "Transformer-Based Model", "cVAE without BN", "cVAE with BN"])
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
        "chemberta": True,
        "chemberta_mean": False,
        "noise": False,
        "grad_clip": False
    }

    df = load_df_w_features()
    if suggest == "MLP":
        model = test_MLP(config)
    if suggest == "MLP with Residual Connections":
        model = ResMLP(config)
    if suggest == "MLP with Residual Connections and BN":
        model = ResMLP_BN(config)
    if suggest == "Transformer-Based Model":
        model = CustomTransformer_v3(config)
    if suggest == "cVAE without BN":
        model = cVAE(config, False)
    if suggest == "cVAE with BN":
        model = cVAE(config, True)
    model = test_MLP(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)
    if suggest == "cVAE without BN" or suggest == "cVAE with BN":
        criterion = cVAE_Loss(0.5158699762912065, 0.17759311956146973, 0.48735628809909753, 0.4192700658080418, 0.510095994098853)
    else:
        criterion = CustomLoss(0.5158699762912065, 0.17759311956146973, 0.48735628809909753, 0.4192700658080418, 0.510095994098853)
    score = leave_one_out_cv(df, model, optimizer, lr_scheduler, criterion, config)
    return score


def model_study(BASE_PATH="optuna/model_study.sqlite3"):
    search_space = {
        "Model": ["MLP", "MLP with Residual Connections", "MLP with Residual Connections and BN", "Transformer-Based Model", "cVAE without BN", "cVAE with BN"]
    }
    ABSOLUTE_PATH = "sqlite:///" + os.path.join(os.path.dirname(__file__), BASE_PATH)
    study = optuna.create_study(
        storage=ABSOLUTE_PATH,
        sampler=optuna.samplers.GridSampler(search_space),
        study_name="Model Selection"
    )
    study.optimize(model_objective)
    print(f"Best value: {study.best_value} (params: {study.best_params})")


def hyper_objective(trial):
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
        "chemberta": True,
        "chemberta_mean": False,
        "noise": False,
        "grad_clip": False
    }
    df = load_df_w_features()
    model = ResMLP(config)
    learning_rate = trial.suggest_loguniform("Learning Rate", 1e-5, 1e-1)
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    mult = trial.suggest_int("Cycle Multiplier", 1,3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, mult)
    criterion = CustomLoss(0.5158699762912065, 0.17759311956146973, 0.48735628809909753, 0.4192700658080418, 0.510095994098853)
    score = leave_one_out_cv(df, model, optimizer, lr_scheduler, criterion, config)
    return score


def hyper_study(BASE_PATH="optuna/hyper_study.sqlite3"):
    ABSOLUTE_PATH = "sqlite:///" + os.path.join(os.path.dirname(__file__), BASE_PATH)
    study = optuna.create_study(
        storage=ABSOLUTE_PATH,
        sampler=optuna.samplers.TPESampler(),
        study_name="Hyperparameter Tuning - ResMLP"
    )
    study.optimize(hyper_objective, n_trials=30)
    print(f"Best value: {study.best_value} (params: {study.best_params})")


def res_objective(trial):
    grad_clip = trial.suggest_categorical("Clip Gradients to 1", [True, False])
    noise = trial.suggest_categorical("Add Noisy Data", [True, False])
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
        "chemberta": True,
        "chemberta_mean": False,
        "noise": noise,
        "grad_clip": grad_clip
    }

    df = load_df_w_features()
    model = ResMLP(config)
    optimizer = torch.optim.AdamW(model.parameters(), 2e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)
    criterion = CustomLoss(0.5158699762912065, 0.17759311956146973, 0.48735628809909753, 0.4192700658080418, 0.510095994098853)
    score = leave_one_out_cv(df, model, optimizer, lr_scheduler, criterion, config)
    return score


def res_study(BASE_PATH="optuna/hyper_study.sqlite3"):
    search_space = {
        "Clip Gradients to 1": [True, False],
        "Add Noisy Data": [True, False]
    }
    ABSOLUTE_PATH = "sqlite:///" + os.path.join(os.path.dirname(__file__), BASE_PATH)
    study = optuna.create_study(
        storage=ABSOLUTE_PATH,
        sampler=optuna.samplers.GridSampler(search_space),
        study_name="Additional Techniques - ResMLP"
    )
    study.optimize(model_objective)
    print(f"Best value: {study.best_value} (params: {study.best_params})")


def cvae_objective(trial):
    noise = trial.suggest_categorical("Add Noisy Data", [True, False])
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
        "chemberta": True,
        "chemberta_mean": False,
        "noise": noise,
        "grad_clip": True
    }

    df = load_df_w_features()
    model = cVAE(config, False)
    optimizer = torch.optim.AdamW(model.parameters(), 2e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)
    criterion = cVAE_Loss(0.5158699762912065, 0.17759311956146973, 0.48735628809909753, 0.4192700658080418)
    score = leave_one_out_cv(df, model, optimizer, lr_scheduler, criterion, config)
    return score


def res_study(BASE_PATH="optuna/hyper_study.sqlite3"):
    search_space = {
        "Add Noisy Data": [True, False]
    }
    ABSOLUTE_PATH = "sqlite:///" + os.path.join(os.path.dirname(__file__), BASE_PATH)
    study = optuna.create_study(
        storage=ABSOLUTE_PATH,
        sampler=optuna.samplers.GridSampler(search_space),
        study_name="Add Noise - cVAE"
    )
    study.optimize(model_objective)
    print(f"Best value: {study.best_value} (params: {study.best_params})")


if __name__ == "__main__":
    chemberta_study()