from rdkit import Chem
from torch_geometric.data import Data, Batch
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

MAX_TOKENS = 121
PAD_TOKEN = 33


def getParquetData(BASE_PATH="data/de_train.parquet"):
    # Read cell_type, SMILES, and gene (target) data from de_train.parquet
    print("Loading Data...")
    ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), BASE_PATH)
    data = pd.read_parquet(ABSOLUTE_PATH, engine="fastparquet")
    cell_types = np.squeeze(data.loc[:, ["cell_type"]].to_numpy())
    smiles = np.squeeze(data.loc[:, ["SMILES"]].to_numpy())
    targets = data.iloc[:, 5:18216].to_numpy()

    # Collect information from compounds, map cell types to integers, and tokenize SMILES
    print("Collecting Features...")
    type_to_num_vect = np.vectorize(type_to_num)
    cell_types = type_to_num_vect(cell_types)
    compound_adjacency_matrices = np.array(smiles_to_adjacency(smiles))
    compound_atom_features = np.array(smiles_to_atom_features(smiles))
    smiles_tokens = np.array(
        [smiles_to_indices(smiles_string) for smiles_string in smiles]
    )
    return (
        cell_types,
        compound_adjacency_matrices,
        compound_atom_features,
        smiles_tokens,
        targets,
    )


class CompoundEncoderDataset(Dataset):
    def __init__(self, BASE_PATH="data/de_train.parquet", device=torch.device("mps")):
        ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), BASE_PATH)
        data = pd.read_parquet(ABSOLUTE_PATH, engine="fastparquet")
        smiles = list(set(data["SMILES"].tolist()))  # Removes duplicates
        graphs = smiles_to_graph(smiles, device=device)
        tokens = smiles_to_indices(smiles, device=device)
        self.data = list(zip(graphs, tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def compound_collate_fn(list_of_tuples):
    """
    Parameters:
    list_of_tuples: list of tuples of (Pyg Graph, Tokens List)

    Output: Pyg Graph Batch, Batched Tokens Tensor
    """
    two_lists = list(map(list, zip(*list_of_tuples)))
    batched_graphs = Batch.from_data_list(two_lists[0])
    batched_tokens = torch.stack(two_lists[1])
    return batched_graphs, batched_tokens


def type_to_num(cell_type):
    CELL_TYPE_MAP = {
        "NK cells": 0,
        "T cells CD4+": 1,
        "T cells CD8+": 2,
        "T regulatory cells": 3,
        "B cells": 4,
        "Myeloid cells": 5,
    }
    return CELL_TYPE_MAP[cell_type]


def smiles_to_graph(smiles_list, device):
    NODE_FEATURE_MAP = {
        "atomic_num": list(range(0, 119)),
        "chirality": [
            "CHI_UNSPECIFIED",
            "CHI_TETRAHEDRAL_CW",
            "CHI_TETRAHEDRAL_CCW",
            "CHI_OTHER",
            "CHI_TETRAHEDRAL",
            "CHI_ALLENE",
            "CHI_SQUAREPLANAR",
            "CHI_TRIGONALBIPYRAMIDAL",
            "CHI_OCTAHEDRAL",
        ],
        "degree": list(range(0, 11)),
        "formal_charge": list(range(-5, 7)),
        "num_hs": list(range(0, 9)),
        "num_radical_electrons": list(range(0, 5)),
        "hybridization": [
            "UNSPECIFIED",
            "S",
            "SP",
            "SP2",
            "SP3",
            "SP3D",
            "SP3D2",
            "OTHER",
        ],
        "is_aromatic": [False, True],
        "is_in_ring": [False, True],
    }

    EDGE_FEATURE_MAP = {
        "bond_type": [
            "UNSPECIFIED",
            "SINGLE",
            "DOUBLE",
            "TRIPLE",
            "QUADRUPLE",
            "QUINTUPLE",
            "HEXTUPLE",
            "ONEANDAHALF",
            "TWOANDAHALF",
            "THREEANDAHALF",
            "FOURANDAHALF",
            "FIVEANDAHALF",
            "AROMATIC",
            "IONIC",
            "HYDROGEN",
            "THREECENTER",
            "DATIVEONE",
            "DATIVE",
            "DATIVEL",
            "DATIVER",
            "OTHER",
            "ZERO",
        ],
        "stereo": [
            "STEREONONE",
            "STEREOANY",
            "STEREOZ",
            "STEREOE",
            "STEREOCIS",
            "STEREOTRANS",
        ],
        "is_conjugated": [False, True],
    }

    graph_list = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)

        xs = []
        for atom in mol.GetAtoms():
            x = []
            x.append(NODE_FEATURE_MAP["atomic_num"].index(atom.GetAtomicNum()))
            x.append(NODE_FEATURE_MAP["chirality"].index(str(atom.GetChiralTag())))
            x.append(NODE_FEATURE_MAP["degree"].index(atom.GetTotalDegree()))
            x.append(NODE_FEATURE_MAP["formal_charge"].index(atom.GetFormalCharge()))
            x.append(NODE_FEATURE_MAP["num_hs"].index(atom.GetTotalNumHs()))
            x.append(
                NODE_FEATURE_MAP["num_radical_electrons"].index(
                    atom.GetNumRadicalElectrons()
                )
            )
            x.append(
                NODE_FEATURE_MAP["hybridization"].index(str(atom.GetHybridization()))
            )
            x.append(NODE_FEATURE_MAP["is_aromatic"].index(atom.GetIsAromatic()))
            x.append(NODE_FEATURE_MAP["is_in_ring"].index(atom.IsInRing()))
            xs.append(x)
        x = torch.tensor(xs, dtype=torch.float32).view(-1, 9)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            e = []
            e.append(EDGE_FEATURE_MAP["bond_type"].index(str(bond.GetBondType())))
            e.append(EDGE_FEATURE_MAP["stereo"].index(str(bond.GetStereo())))
            e.append(EDGE_FEATURE_MAP["is_conjugated"].index(bond.GetIsConjugated()))
            edge_indices += [[i, j], [j, i]]
            edge_attrs += [e, e]
        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).view(-1, 3)
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        graph_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, device=device))
    return graph_list


def smiles_to_indices(smiles_stack, device, max_tokens=MAX_TOKENS):
    CUSTOM_TOKENS = {
        "C": 0,
        "c": 1,
        "N": 2,
        "n": 3,
        "O": 4,
        "o": 5,
        "S": 6,
        "s": 7,
        "F": 8,
        "H": 9,
        "Cl": 10,
        "Br": 11,
        "I": 12,
        "B": 13,
        "(": 14,
        ")": 15,
        "[": 16,
        "]": 17,
        "-": 18,
        "+": 19,
        "=": 20,
        "/": 21,
        "\\": 22,
        "#": 23,
        "@": 24,
        "@@": 25,
        "1": 26,
        "2": 27,
        "3": 28,
        "4": 29,
        "5": 30,
        "6": 31,
        "7": 32,
    }
    smiles_tokens = []
    for smiles_string in smiles_stack:
        tokens = []
        i = 0
        while i < len(smiles_string):
            # Check for two character tokens (Br, Cl, @@)
            if smiles_string[i : i + 2] in CUSTOM_TOKENS:
                tokens.append(CUSTOM_TOKENS[smiles_string[i : i + 2]])
                i += 2
            # Check for single-character tokens
            elif smiles_string[i] in CUSTOM_TOKENS:
                tokens.append(CUSTOM_TOKENS[smiles_string[i]])
                i += 1
            else:
                raise ValueError(f"Invalid atom: {smiles_string[i]}")
        tokens += [PAD_TOKEN] * (max_tokens - len(tokens))
        smiles_tokens.append(torch.as_tensor(tokens, device=device))
    return smiles_tokens


def observe_data(BASE_PATH="data/de_train.parquet"):
    """
    Plots the difference in gene expression between different cell types. Made for
    the purpose of seeing how the effects on cell types we have data for
    (NK cells, T cells CD4+, T cells CD8+, and T regulatory cells) can be used
    to predict the effects on B cells and Myeloid cells.
    """
    ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), BASE_PATH)
    data = pd.read_parquet(ABSOLUTE_PATH, engine="fastparquet")
    bcell_rows = data.loc[data["cell_type"] == "B cells"]
    x_axis = np.arange(18211)
    for _, row in bcell_rows.iterrows():
        figure, axis = plt.subplots(2, 3, sharex=True, sharey=True)
        same_compound = data.loc[data["sm_name"] == row["sm_name"]].reset_index()
        for i, cell in same_compound.iterrows():
            y_axis = cell.iloc[5:18216].to_numpy()
            a, b = i % 2, int(i / 2)
            axis[a, b].plot(x_axis, y_axis)
            axis[a, b].set_xlabel("18,211 Cell Types")
            axis[a, b].set_ylabel("18,211 Cell Types")
            axis[a, b].set_title(cell["cell_type"])
        # plt.title(row["sm_name"])
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.show()


if __name__ == "__main__":
    observe_data()
