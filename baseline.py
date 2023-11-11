import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

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
        
"""
For some reason I was having trouble finding a simple weighted sum model,
so I just made one in pytorch that initializes the weights as an average.
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

def weighted_sum(DATA_PATH="data/de_train.parquet", ID_PATH="data/id_map.csv"):
    ABSOLUTE_DATA_PATH = os.path.join(os.path.dirname(__file__), DATA_PATH)
    data = pd.read_parquet(ABSOLUTE_DATA_PATH, engine="fastparquet")
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

    X = torch.as_tensor(np.array(x, dtype="float32"))
    Y_m = torch.as_tensor(np.array(y_m, dtype="float32"))
    Y_b = torch.as_tensor(np.array(y_b, dtype="float32"))

    m_model = WeightedSum()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(m_model.parameters(), 1e-3)
    for i in range(1000):
        optimizer.zero_grad()
        outputs = m_model(X)
        loss = criterion(outputs, Y_m)
        loss.backward()
        optimizer.step()
        print(loss.item())
    

    """
    b_model = WeightedSum()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(b_model.parameters(), 1e3)
    for i in range(1000):
        optimizer.zero_grad()
        outputs = m_model(X)
        loss = criterion(outputs, Y_b)
        loss.backward()
        optimizer.step()
        print(loss.item())
    """

if __name__ == "__main__":
    weighted_sum()