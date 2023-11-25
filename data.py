import os
import pandas as pd
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, Batch
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA

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


class AutoencoderDataset(Dataset):
    def __init__(self, BASE_PATH="data/de_train.parquet", device=torch.device("mps")):
        ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), BASE_PATH)
        data = pd.read_parquet(ABSOLUTE_PATH, engine="fastparquet")
        self.data = torch.as_tensor(
            data.iloc[:, 5:18216].to_numpy(dtype="float32"), device=device
        )

    def __len__(self):
        return self.data.shape[0]

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

        graph_list.append(
            Data(x=x, edge_index=edge_index, edge_attr=edge_attr, device=device)
        )
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


def cell_correlation_heatmaps(BASE_PATH="data/de_train.parquet", sortby="cell_type"):
    ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), BASE_PATH)
    data = pd.read_parquet(ABSOLUTE_PATH, engine="fastparquet")
    # sort by cell type
    data.sort_values(by=sortby, inplace=True)
    # gene expression data for each cell
    single_cell_vectors = data.iloc[:, 5:18216].to_numpy()
    # correlation between vectors is the cosine between them
    dot_products = single_cell_vectors @ single_cell_vectors.T
    norms = np.sqrt(np.sum(single_cell_vectors**2, axis=1))
    corr_mat = dot_products / (norms[:, np.newaxis] * norms[np.newaxis, :])
    # create string maps for readability
    compounds = list(dict.fromkeys(data["sm_name"].tolist()))
    COMPOUND_MAP = {name: compounds.index(name) for name in compounds}
    CELL_MAP = {
        "B cells": "B",
        "Myeloid cells": "M",
        "NK cells": "NK",
        "T cells CD8+": "T8",
        "T cells CD4+": "T4",
        "T regulatory cells": "TR",
    }
    # create column/row labels
    labels = []
    for _, row in data.iterrows():
        compound, cell_type = COMPOUND_MAP[row["sm_name"]], CELL_MAP[row["cell_type"]]
        labels.append(f"{compound}: {cell_type}")
    # dataframe with columns and rows
    corr_df = pd.DataFrame(data=corr_mat, index=labels, columns=labels)
    print(corr_df)
    # mask upper half (correlation matrix is symmetric)
    mask = np.zeros_like(corr_mat, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_df, mask=mask, square=True)
    sns.set(font_scale=0.1)
    # plt.savefig(f"visuals/cells_corr_heatmap_({sortby}).png", bbox_inches="tight", dpi=400)
    plt.show()


def gene_correlation_heatmaps(BASE_PATH="data/de_train.parquet"):
    print("right place")
    ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), BASE_PATH)
    data = pd.read_parquet(ABSOLUTE_PATH, engine="fastparquet")
    # each vector representing an individual gene
    single_gene_vectors = data.iloc[:, 5:18216].to_numpy().T
    # correlation between vectors is the cosine between them
    # TODO: do on GPU with torch
    # (18211,614)x(614,18211)=(18211,18211) this is really slow right now
    dot_products = single_gene_vectors @ single_gene_vectors.T
    norms = np.sqrt(np.sum(single_gene_vectors**2, axis=1))
    corr_mat = dot_products / (norms[:, np.newaxis] * norms[np.newaxis, :])
    # mask upper half (correlation matrix is symmetric)
    mask = np.zeros_like(corr_mat, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_mat, mask=mask, square=True, xticklabels=False, yticklabels=False)
    plt.savefig("visuals/genes_heatmap.png", bbox_inches="tight", dpi=400)
    plt.show()


def get_cell_data(BASE_PATH="data/de_train.parquet"):
    ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), BASE_PATH)
    data = pd.read_parquet(ABSOLUTE_PATH, engine="fastparquet")
    data = data.sort_values(by="cell_type").reset_index(drop=True)
    data["index1"] = data.index
    # gene expression data for each cell
    single_cell_vectors = data.iloc[:, 5:18216].to_numpy()
    return data, single_cell_vectors


def get_handles(palette=sns.color_palette("Dark2")):
    labels = [
        "Myeloid cells",
        "B cells",
        "T regulatory cells",
        "T cells CD8+",
        "T cells CD4+",
        "NK cells",
    ]
    handles = [
        Line2D(
            [],
            [],
            color=palette[idx],
            marker=".",
            linestyle="None",
            markersize=10,
            label=label,
        )
        for idx, label in enumerate(labels)
    ]
    return handles


def plot_cell_types_colored(data, embedding, palette=sns.color_palette("Dark2")):
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[
            sns.color_palette("Dark2")[int(x)]
            for x in data.cell_type.map(
                {
                    "NK cells": 5,
                    "T cells CD4+": 4,
                    "T cells CD8+": 3,
                    "T regulatory cells": 2,
                    "B cells": 1,
                    "Myeloid cells": 0,
                }
            )
        ],
    )


def arrows(data, embedding):
    # iterating through b cells
    bcell_rows = data.loc[data["cell_type"] == "B cells"]
    for _, bcell in bcell_rows.iterrows():
        # certain compound of this b cell
        compound = bcell["sm_name"]
        # all other cells with the same compound
        cells = data.loc[data["sm_name"] == compound]
        # myeloid cell with this compound
        myeloid = cells.loc[cells["cell_type"] == "Myeloid cells"]
        # embedding coordinates of b cell and myeloid cell
        bcell_coords = embedding[bcell.values[-1], :]
        myeloid_coords = embedding[myeloid.values[0][-1], :]
        # iterating through all cells of this compound
        for _, compound_cell in cells.iterrows():
            # only draw arrows from non b/myeloid cells
            if compound_cell["cell_type"] not in ["Myeloid cells", "B cells"]:
                # collect embedding coordinates of cell
                index = compound_cell.values[-1]
                x, y = embedding[index, 0], embedding[index, 1]

                dx_b, dy_b = bcell_coords[0] - x, bcell_coords[1] - y
                dx_m, dy_m = myeloid_coords[0] - x, myeloid_coords[1] - y

                plt.arrow(x, y, dx_b, dy_b, width=0.00001)
                plt.arrow(x, y, dx_m, dy_m, width=0.00001)


def umap_cells(metric, BASE_PATH="data/de_train.parquet", arrows=False):
    data, single_cell_vectors = get_cell_data()
    embedding = umap.UMAP(n_components=2, metric=metric).fit_transform(
        single_cell_vectors
    )
    # plot UMAP embedding colored by cell type
    plot_cell_types_colored(data, embedding)

    # draw arrows pointing to b cells and myeloid cells from other cell types
    if arrows:
        arrows(data, embedding)

    plt.legend(handles=get_handles())
    plt.title(f"UMAP Across Cells, {metric} metric")
    plt.savefig(f"visuals/cells_umap_{metric}.png", bbox_inches="tight", dpi=400)
    # plt.show()


def tsne_cells(metric, perplexity=30, BASE_PATH="data/de_train.parquet", arrows=False):
    data, single_cell_vectors = get_cell_data()
    embedding = TSNE(
        n_components=2, metric=metric, perplexity=perplexity
    ).fit_transform(single_cell_vectors)
    # plot TSNE embedding colored by cell type
    plot_cell_types_colored(data, embedding)

    # draw arrows pointing to b cells and myeloid cells from other cell types
    if arrows:
        arrows(data, embedding)

    plt.legend(handles=get_handles())
    plt.title(f"t-SNE Across Cells, {metric} metric, {perplexity} perplex.")
    plt.savefig(f"visuals/cells_tsne_{metric}.png", bbox_inches="tight", dpi=400)
    # plt.show()


def kernel_pca_cells(kernel="rbf", BASE_PATH="data/de_train.parquet", arrows=False):
    data, single_cell_vectors = get_cell_data()
    embedding = KernelPCA(n_components=2, kernel=kernel).fit_transform(
        single_cell_vectors
    )
    # plot kernel pca embedding colored by cell type
    plot_cell_types_colored(data, embedding)

    # draw arrows pointing to b cells and myeloid cells from other cell types
    if arrows:
        arrows(data, embedding)

    plt.legend(handles=get_handles())
    plt.title(f"{kernel} PCA Across Cells")
    plt.savefig(f"visuals/cells_{kernel}_pca.png", bbox_inches="tight", dpi=400)
    # plt.show()


if __name__ == "__main__":
    """
    Data Visualization so far. Uncomment the lines of code to use.

    Note that with the heatmaps, the tick marks aren't very useful when they are there,
    because there are too many. Instead, it's easier just to see if there's a general
    pattern, like if certain cell types are correlated with others.

    My umap and tsne functions have an "arrow" boolean parameter, which will draw
    arrows between T and NK cells to Myeloid and B cells if arrow is True. Right now
    it's not helping, it just clutters things. Maybe some formatting would help.
    """

    """
    Heatmap for seeing if there are correlations between cell types (single cell heatmap)
    """
    # cell_correlation_heatmaps(sortby="cell_type")

    """
    Heatmap for seeing if there are correlations between compounds (single cell heatmap)
    """
    # cell_correlation_heatmaps(sortby="sm_name")

    """
    Heatmap for seeing if there are correlations between genes (single gene heatmap)
    """
    # gene_correlation_heatmaps()

    """
    UMAP across cells, cell types colored
    Cosine and correlation metrics seem to work better
    """
    # umap_cells(metric="cosine")

    """
    t-SNE across cells, cell types colored
    Cosine metric seem to work better
    """
    # tsne_cells(metric="cosine", perplexity=30)

    """
    Kernel PCA across cells, cell types colored
    """
    kernel_pca_cells(kernel="rbf")
