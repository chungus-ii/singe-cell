import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os, csv, datetime


def unweighted_average(DATA_PATH="data/de_train.parquet", ID_PATH="data/id_map.csv"):
    ABSOLUTE_DATA_PATH = os.path.join(os.path.dirname(__file__), DATA_PATH)
    data = pd.read_parquet(ABSOLUTE_DATA_PATH, engine="fastparquet")
    ABSOLUTE_ID_PATH = os.path.join(os.path.dirname(__file__), ID_PATH)
    testing_pairs = pd.read_csv(ABSOLUTE_ID_PATH)
    results = []
    for _, row in testing_pairs.iterrows():
        compound = row["sm_name"]
        average = data.loc[data['sm_name'] == compound].to_numpy()[:, 5:18216].mean(axis=0)
        results.append(average)
    results = pd.DataFrame(results)
    SUBMISSION_PATH = os.path.join(os.path.dirname(__file__), "submissions/avg_baseline_submission.csv")
    columns = data.columns[5:18216]
    results.columns = columns
    results.index.name = "id"
    results.to_csv(SUBMISSION_PATH)
    FILE_PATH = "submissions/submission_{date:%Y-%m-%d_%H:%M:%S}.csv".format(date=datetime.datetime.now())
    ABSOLUTE_DATA_PATH = os.path.join(os.path.dirname(__file__), DATA_PATH)


"""
For some reason I was having trouble finding these simple models implemented,
so I just made them in pytorch and initialized the weights so that they start as averages.
There is probably a better way to do this.
"""
class WeightedSum(nn.Module):
    def __init__(self):
        super(WeightedSum, self).__init__()
        self.w1 = nn.Parameter(torch.tensor(1/3))
        self.w2 = nn.Parameter(torch.tensor(1/3))
        self.w3 = nn.Parameter(torch.tensor(1/3))

    def forward(self, x):
        x1, x2, x3 = x.chunk(3, dim=-1)
        output = (self.w1 * x1) + (self.w2 * x2) + (self.w3 * x3)
        return output


class ScalarWeights(nn.Module):
    def __init__(self):
        super(ScalarWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor(1/3))
        self.w2 = nn.Parameter(torch.tensor(1/3))
        self.w3 = nn.Parameter(torch.tensor(1/3))
        self.b = nn.Parameter(torch.zeros(18211))

    def forward(self, x):
        x1, x2, x3 = x.chunk(3, dim=-1)
        output = (self.w1 * x1) + (self.w2 * x2) + (self.w3 * x3) + self.b
        return output


class VectorWeights(nn.Module):
    def __init__(self):
        super(VectorWeights, self).__init__()
        self.w1 = nn.Parameter((1/3)*torch.ones(18211))
        self.w2 = nn.Parameter((1/3)*torch.ones(18211))
        self.w3 = nn.Parameter((1/3)*torch.ones(18211))
        self.b = nn.Parameter(torch.zeros(18211))

    def forward(self, x):
        x1, x2, x3 = x.chunk(3, dim=-1)
        output = (self.w1 * x1) + (self.w2 * x2) + (self.w3 * x3) + self.b
        return output


def train(model, X, Y, epochs=1000, lr=1e-2):
    model.train()
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    for i in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        print(loss.item())
    return model


def train_2_models(DATA_PATH="data/de_train.parquet", model=VectorWeights, epochs=10000, lr=1e-2, device=torch.device("cpu")):
    ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), DATA_PATH)
    data = pd.read_parquet("/Users/iishaan/all-code/kaggle/single-cell/data/de_train.parquet", engine="fastparquet")
    bdata = data.loc[data["cell_type"] == "B cells"]
    compounds = list(set(bdata["sm_name"].tolist()))
    x = []
    y_m, y_b = [], []
    for compound in compounds:
        rows = data.loc[data["sm_name"] == compound]
        b_cell = rows.loc[rows["cell_type"] == "B cells"].iloc[0, 5:18216].to_numpy()
        m_cell = rows.loc[rows["cell_type"] == "Myeloid cells"].iloc[0, 5:18216].to_numpy()
        t_c4 = rows.loc[rows["cell_type"] == "T cells CD4+"].iloc[0, 5:18216].to_numpy()
        t_r = rows.loc[rows["cell_type"] == "T regulatory cells"].iloc[0, 5:18216].to_numpy()
        nk = rows.loc[rows["cell_type"] == "NK cells"].iloc[0, 5:18216].to_numpy()
        x.append(np.concatenate((t_c4, t_r, nk)))
        y_m.append(m_cell)
        y_b.append(b_cell)

    X = torch.as_tensor(np.array(x, dtype="float32")).to(device)
    Y_m = torch.as_tensor(np.array(y_m, dtype="float32")).to(device)
    Y_b = torch.as_tensor(np.array(y_b, dtype="float32")).to(device)

    m_model = model().to(device)
    b_model = model().to(device)

    m_model = train(m_model, X, Y_m, epochs, lr)
    b_model = train(b_model, X, Y_b, epochs, lr)

    return m_model, b_model


def make_predictions(m_model, b_model, FILE_PATH, DATA_PATH="data/de_train.parquet", ID_PATH="data/id_map.csv"):
    m_model = m_model.eval().to(torch.device("cpu"))
    b_model = b_model.eval().to(torch.device("cpu"))
    ABSOLUTE_DATA_PATH = os.path.join(os.path.dirname(__file__), DATA_PATH)
    data = pd.read_parquet(ABSOLUTE_DATA_PATH, engine="fastparquet")
    ABSOLUTE_ID_PATH = os.path.join(os.path.dirname(__file__), ID_PATH)
    pairs = pd.read_csv(ABSOLUTE_ID_PATH)
    rows = []
    for i, pair in pairs.iterrows():
        compound = pair["sm_name"]
        cell_type = pair["cell_type"]
        cells = data.loc[data["sm_name"] == compound]
        t_c4 = cells.loc[cells["cell_type"] == "T cells CD4+"].iloc[0, 5:18216].to_numpy(dtype="float32")
        t_r = cells.loc[cells["cell_type"] == "T regulatory cells"].iloc[0, 5:18216].to_numpy(dtype="float32")
        nk = cells.loc[cells["cell_type"] == "NK cells"].iloc[0, 5:18216].to_numpy(dtype="float32")
        x = torch.as_tensor(np.concatenate((t_c4, t_r, nk)))
        if cell_type == "B cells":
            output = b_model(x)
        elif cell_type == "Myeloid cells":
            output = m_model(x)
        rows.append([i] + output.tolist())
    columns = ["id"] + list(data.columns)[5:18216]
    ABSOLUTE_FILE_PATH = os.path.join(os.path.dirname(__file__), FILE_PATH)
    print(ABSOLUTE_FILE_PATH)
    with open(ABSOLUTE_FILE_PATH, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(columns)
        csvwriter.writerows(rows)


if __name__ == "__main__":
    # Average Model: 0.627
    m_model, b_model = train_2_models(model=ScalarWeights, epochs=1500, lr=6e-2, device=torch.device("mps"))
    make_predictions(m_model, b_model, FILE_PATH="submissions/MAE_scalar_w_baseline_submission.csv")