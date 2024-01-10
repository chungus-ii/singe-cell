import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from data_sc import load_df_w_features, calculate_dim, tscore_to_de, TrainDataset, ValDataset
from train_sc import leave_one_out_cv, CustomLoss, mrrmse
from model_sc import test_MLP, CustomTransformer_v3, ResMLP, ResMLP_BN, cVAE

device = torch.device("cpu")

config_main = {
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
config_main_clip = {
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
    "grad_clip": True
}
config_full = {
    "emb_cell": True,
    "emb_sm": True,
    "mean_cell": True,
    "mean_sm": True,
    "std_cell": True,
    "std_sm": True,
    "median_cell": True,
    "median_sm": True,
    "control": True,
    "t_score": False,
    "svd": True,
    "pca": False,
    "inv_transform": True,
    "chemberta": True,
    "chemberta_mean": True,
    "noise": False,
    "grad_clip": False
}
config_full_clip = {
    "emb_cell": True,
    "emb_sm": True,
    "mean_cell": True,
    "mean_sm": True,
    "std_cell": True,
    "std_sm": True,
    "median_cell": True,
    "median_sm": True,
    "control": False,
    "t_score": True,
    "svd": True,
    "pca": False,
    "inv_transform": True,
    "chemberta": True,
    "chemberta_mean": True,
    "noise": False,
    "grad_clip": True
}
config_full_no_svd = {
    "emb_cell": True,
    "emb_sm": True,
    "mean_cell": True,
    "mean_sm": True,
    "std_cell": True,
    "std_sm": True,
    "median_cell": True,
    "median_sm": True,
    "control": False,
    "t_score": True,
    "svd": False,
    "pca": False,
    "inv_transform": False,
    "chemberta": True,
    "chemberta_mean": True,
    "noise": False,
    "grad_clip": False
}
config_full_no_svd_clip = {
    "emb_cell": True,
    "emb_sm": True,
    "mean_cell": True,
    "mean_sm": True,
    "std_cell": True,
    "std_sm": True,
    "median_cell": True,
    "median_sm": True,
    "control": False,
    "t_score": True,
    "svd": False,
    "pca": False,
    "inv_transform": False,
    "chemberta": True,
    "chemberta_mean": True,
    "noise": False,
    "grad_clip": True
}

def save_weights(df, config, BASE_PATH):
    model = ResMLP(config)
    optimizer = torch.optim.AdamW(model.parameters(), 2e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)
    criterion = CustomLoss(0.5158699762912065, 0.17759311956146973, 0.48735628809909753, 0.4192700658080418, 0.510095994098853)
    score, weights = leave_one_out_cv(df, model, optimizer, lr_scheduler, criterion, config, weights=True)
    print(f"final score: {score}")
    # BASE_PATH = "weights/main/model_"
    ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), BASE_PATH)
    for i, single_model_weights in enumerate(weights):
        torch.save(single_model_weights, ABSOLUTE_PATH + f"{i}.pth")
    return score

def save_models():
    df = load_df_w_features()
    scores = []
    scores.append(save_weights(df, config_main, "weights/main_3/model_"))
    """
    scores.append(save_weights(df, config_main_clip, "weights/main_clip/model_"))
    scores.append(save_weights(df, config_full, "weights/full/model_"))
    scores.append(save_weights(df, config_full_clip, "weights/full_clip/model_"))
    scores.append(save_weights(df, config_full_no_svd, "weights/full_no_svd/model_"))
    scores.append(save_weights(df, config_full_no_svd_clip, "weights/full_no_svd_clip/model_"))
    """
    print(scores)


def check_val(df, config, BASE_PATH):
    # BASE_PATH = "weights/main/model_"
    ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), BASE_PATH)
    b_cells = df.loc[df["cell_type"] == "B cells"]
    myeloid_cells = df.loc[df["cell_type"] == "Myeloid cells"]
    other_cells = df.loc[
        (df["cell_type"] != "Myeloid cells") & (df["cell_type"] != "B cells")
    ]

    score = 0
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
        val_dataloader = DataLoader(val_dataset, batch_size=2)

        weights = torch.load(ABSOLUTE_PATH + f"{i}.pth", map_location=device)
        model = ResMLP(config)
        model.load_state_dict(weights)
        model.eval()

        val_loss = 0
        for x in val_dataloader:
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
        score += val_loss
    score /= len(b_cells)
    return score


def check_all_models():
    df = load_df_w_features()
    score = check_val(df, config_main, "weights/main_3/model_")
    print(f"main model score: {score}")
    """
    score = check_val(df, config_main, "weights/main/model_")
    print(f"main model score: {score}")

    score = check_val(df, config_main_clip, "weights/main_clip/model_")
    print(f"main model clip score: {score}")

    score = check_val(df, config_full, "weights/full/model_")
    print(f"full model score: {score}")

    score = check_val(df, config_full_clip, "weights/full_clip/model_")
    print(f"full model clip score: {score}")

    score = check_val(df, config_full_no_svd, "weights/full_no_svd/model_")
    print(f"full model no svd score: {score}")

    score = check_val(df, config_full_no_svd_clip, "weights/full_no_svd_clip/model_")
    print(f"full model no svd clip score: {score}")
    """


def create_submission(df, config, BASE_PATH):
    ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), BASE_PATH)
    b_cells = df.loc[df["cell_type"] == "B cells"]
    myeloid_cells = df.loc[df["cell_type"] == "Myeloid cells"]
    other_cells = df.loc[
        (df["cell_type"] != "Myeloid cells") & (df["cell_type"] != "B cells")
    ]
    models = []
    datasets = []
    dfs = []
    for i in range(len(b_cells)):
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
        dfs.append(train_df)
        train_dataset = TrainDataset(train_df, config, device)
        datasets.append(train_dataset)

        weights = torch.load(ABSOLUTE_PATH + f"{i}.pth", map_location=device)
        model = ResMLP(config)
        model.load_state_dict(weights)
        model.eval()
        models.append(model)

    ABSOLUTE_ID_PATH = os.path.join(os.path.dirname(__file__), "data/id_map.csv")
    pairs = pd.read_csv(ABSOLUTE_ID_PATH)
    for i, pair in pairs.iterrows():
        cell_type = pair["cell_type"]
        sm_name = pair["sm_name"]
        
        for model, dataset, df in zip(models, datasets, dfs):
            cell_info_example_idx = df.loc[df["cell_type"] == cell_type].index[0]
            cell_info_example = dataset[cell_info_example_idx]
            sm_name_info_example_idx = df.loc[df["sm_name"] == sm_name].index[0]
            sm_name_info_example = dataset[sm_name_info_example_idx]




if __name__ == "__main__":
    save_models()
    check_all_models()