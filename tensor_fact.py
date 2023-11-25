import torch
import numpy as np
import tensorly as tl
import pandas as pd
import os, csv


def get_tensor(device=torch.device("mps"), DATA_PATH="data/de_train.parquet"):
    ABSOLUTE_DATA_PATH = os.path.join(os.path.dirname(__file__), DATA_PATH)
    data = pd.read_parquet(ABSOLUTE_DATA_PATH, engine="fastparquet")
    all_compound_names = list(set(data["sm_name"].tolist()))
    COMPOUND_MAP = {compound: i for (i, compound) in enumerate(all_compound_names)}
    CELL_TYPE_MAP = {
        "NK cells": 0,
        "T cells CD4+": 1,
        "T cells CD8+": 2,
        "T regulatory cells": 3,
        "B cells": 4,
        "Myeloid cells": 5,
    }
    to_fill = torch.zeros((15, len(all_compound_names), 18211), dtype=torch.float32)
    for i, row in data.iterrows():
        compound_idx = COMPOUND_MAP[row["sm_name"]]
        cell_type_idx = CELL_TYPE_MAP[row["cell_type"]]
        to_fill[cell_type_idx, compound_idx, :] = torch.tensor(row.iloc[5:18216].to_numpy(dtype="float32"))
    mask = (to_fill != 0).int()
    columns = ["id"] + list(data.columns)[5:18216]
    return to_fill.to(device), mask.to(device), (COMPOUND_MAP, CELL_TYPE_MAP), columns


def make_predictions(COMPOUND_MAP, CELL_TYPE_MAP, complete_tensor, columns, SUBMISSION_PATH, ID_PATH="data/id_map.csv"):
    ABSOLUTE_ID_PATH = os.path.join(os.path.dirname(__file__), ID_PATH)
    pairs = pd.read_csv(ABSOLUTE_ID_PATH)
    rows = []
    for i, pair in pairs.iterrows():
        compound_idx = COMPOUND_MAP[pair["sm_name"]]
        cell_type_idx = CELL_TYPE_MAP[pair["cell_type"]]
        rows.append([i] + complete_tensor[cell_type_idx,compound_idx, :].tolist())
    ABSOLUTE_SUBMISSION_PATH = os.path.join(os.path.dirname(__file__), SUBMISSION_PATH)
    with open(ABSOLUTE_SUBMISSION_PATH, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(columns)
        csvwriter.writerows(rows)


def cp_decomp(uncomplete_tensor, mask, n_iter_max=2000, rank=8):
    tl.set_backend("pytorch")
    weights, factors = tl.decomposition.parafac(uncomplete_tensor, rank=rank, mask=mask)
    cp_reconstruction = tl.cp_to_tensor((weights, factors))
    return cp_reconstruction


def tucker_decomp(uncomplete_tensor, mask, n_iter_max=2000, rank=[2, 10, 100]):
    tl.set_backend("pytorch")
    core, tucker_factors = tl.decomposition.tucker(uncomplete_tensor, rank=rank, mask=mask)
    tucker_reconstruction = tl.tucker_to_tensor((core, tucker_factors))
    return tucker_reconstruction


if __name__ == "__main__":
    # [2, 10, 100] optimal?
    rank = [2, 10, 25]
    uncomplete_tensor, mask, maps, columns = get_tensor()
    print("got data")
    complete_tensor = tucker_decomp(uncomplete_tensor, mask, rank=rank)
    print("completed tensor decomp, now making predictions")
    make_predictions(*maps, complete_tensor, columns, f"submissions/tucker_decomp_submission.csv")