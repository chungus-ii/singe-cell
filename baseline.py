import torch
import pandas as pd
import numpy as np
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
        

if __name__ == "__main__":
    unweighted_average()