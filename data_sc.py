import os
import pandas as pd
import numpy as np
import time
from scipy.special import stdtrit, stdtr
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import DBSCAN, KMeans
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer


def save_chemberta_embeddings(train=True, SMILES=None):
    if train:
        df = load_df()
        smiles_list = df["SMILES"].tolist()
    else:
        smiles_list = SMILES

    chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    chemberta.eval()

    embeddings = torch.zeros(len(smiles_list), 600)
    embeddings_mean = torch.zeros(len(smiles_list), 600)

    with torch.no_grad():
        for i, smiles in enumerate(smiles_list):
            encoded_input = tokenizer(smiles, return_tensors="pt", padding=False, truncation=True)
            model_output = chemberta(**encoded_input)
            embedding = model_output[0][::,0,::]
            embeddings[i] = embedding
            embedding = torch.mean(model_output[0], 1)
            embeddings_mean[i] = embedding
    embeddings = embeddings.numpy()
    embeddings_mean = embeddings_mean.numpy()
    
    if train:
        ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), "data/chemberta_train")
        ABSOLUTE_PATH_MEAN = os.path.join(os.path.dirname(__file__), "data/chemberta_mean_train")
    else:
        ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), "data/chemberta_test")
        ABSOLUTE_PATH_MEAN = os.path.join(os.path.dirname(__file__), "data/chemberta_mean_test")
    np.save(ABSOLUTE_PATH, embeddings)
    np.save(ABSOLUTE_PATH_MEAN, embeddings_mean)


def de_to_tscore(de):
    p_value = np.power(10, -np.abs(de))
    tscore = -stdtrit(420, 0.5 * p_value) * np.sign(de)
    return tscore


def tscore_to_de(t_score):
    p_value = stdtr(420, -np.abs(t_score) * 2)
    p_value = p_value.clip(min=1e-20, max=None)
    de = -np.log10(p_value) * np.sign(t_score)
    return de


def get_cell_sm_one_hot(train_df):
    to_be_encoded = train_df[["cell_type", "sm_name"]]
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(to_be_encoded)
    return encoded


def load_df_w_features(BASE_PATH="data/de_train.parquet"):
    ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), BASE_PATH)
    df = pd.read_parquet(ABSOLUTE_PATH, engine="fastparquet")

    t_scores = de_to_tscore(df.iloc[:,list(range(5, df.shape[1]))])
    encoded = pd.DataFrame(get_cell_sm_one_hot(df))
    df = pd.concat([df, t_scores, encoded], axis=1)

    ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), "data/chemberta_train.npy")
    ABSOLUTE_PATH_MEAN = os.path.join(os.path.dirname(__file__), "data/chemberta_mean_train.npy")
    if os.path.isfile(ABSOLUTE_PATH) and os.path.isfile(ABSOLUTE_PATH_MEAN):
        embeddings = pd.DataFrame(np.load(ABSOLUTE_PATH))
        embedding_means = pd.DataFrame(np.load(ABSOLUTE_PATH_MEAN))
        df = pd.concat([df, embeddings, embedding_means], axis=1)
    else:
        raise Exception("Make sure to create chemberta embeddings with save_chemberta_embeddings()")

    df[["cell_type", "sm_name", "sm_lincs_id", "SMILES"]] = df[["cell_type", "sm_name", "sm_lincs_id", "SMILES"]].astype("category")
    df["control"] = df["control"].astype("int8")
    df.iloc[:,list(range(5, 5 + (18211*2)))] = df.iloc[:,list(range(5, 5 + (18211*2)))].astype("float32")
    df.iloc[:,list(range(5 + (18211*2), 5 + (18211*2) + 152))] = df.iloc[:,list(range(5 + (18211*2), 5 + (18211*2) + 152))].astype("int8")
    df.iloc[:,list(range(5 + (18211*2) + 152, df.shape[1]))] = df.iloc[:,list(range(5 + (18211*2) + 152, df.shape[1]))].astype("float32")
    print(df)
    return df


def load_df(BASE_PATH="data/de_train.parquet"):
    ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), BASE_PATH)
    df = pd.read_parquet(ABSOLUTE_PATH, engine="fastparquet")
    return df


def get_cell_df(train_df):
    cell_df = train_df.iloc[:, [0] + list(range(5, train_df.shape[1]))]
    return cell_df


def get_sm_df(train_df):
    sm_df = train_df.iloc[:, [1] + list(range(5, train_df.shape[1]))]
    return sm_df


def get_cell_means(cell_df):
    cell_means = (
        cell_df.groupby("cell_type", observed=False)
        .mean()
        .reset_index()
        .set_index("cell_type")
        .T.to_dict("list")
    )
    return cell_means


def get_sm_means(sm_df):
    sm_means = (
        sm_df.groupby("sm_name", observed=False)
        .mean()
        .reset_index()
        .set_index("sm_name")
        .T.to_dict("list")
    )
    return sm_means


def get_cell_stds(cell_df):
    cell_stds = (
        cell_df.groupby("cell_type", observed=False)
        .std()
        .reset_index()
        .set_index("cell_type")
        .T.to_dict("list")
    )
    return cell_stds


def get_sm_stds(sm_df):
    sm_stds = (
        sm_df.groupby("sm_name", observed=False)
        .std()
        .reset_index()
        .set_index("sm_name")
        .T.to_dict("list")
    )
    return sm_stds


def get_cell_medians(cell_df):
    cell_medians = (
        cell_df.groupby("cell_type", observed=False)
        .median()
        .reset_index()
        .set_index("cell_type")
        .T.to_dict("list")
    )
    return cell_medians


def get_sm_medians(sm_df):
    sm_medians = (
        sm_df.groupby("sm_name", observed=False)
        .median()
        .reset_index()
        .set_index("sm_name")
        .T.to_dict("list")
    )
    return sm_medians


def is_control(train_df):
    control = train_df[["control"]]
    return control


# for embedding table
def embed_cells(train_df):
    cell_types = train_df[["cell_type"]]
    return cell_types.apply(np.vectorize(cell_to_num))


def cell_to_num(x):
    CELL_TYPE_MAP = {
        "NK cells": 0,
        "T cells CD4+": 1,
        "T cells CD8+": 2,
        "T regulatory cells": 3,
        "B cells": 4,
        "Myeloid cells": 5,
    }
    return CELL_TYPE_MAP[x]


# for embedding table
def embed_sms(train_df):
    sm_names = train_df[["sm_name"]]
    return sm_names.apply(np.vectorize(sm_to_num))


def sm_to_num(x):
    SM_MAP = {
        "Clotrimazole": 0,
        "Mometasone Furoate": 1,
        "Idelalisib": 2,
        "Vandetanib": 3,
        "Bosutinib": 4,
        "Ceritinib": 5,
        "Lamivudine": 6,
        "Crizotinib": 7,
        "Cabozantinib": 8,
        "Flutamide": 9,
        "Dasatinib": 10,
        "Selumetinib": 11,
        "Trametinib": 12,
        "ABT-199 (GDC-0199)": 13,
        "Oxybenzone": 14,
        "Vorinostat": 15,
        "Raloxifene": 16,
        "Linagliptin": 17,
        "Lapatinib": 18,
        "Canertinib": 19,
        "Disulfiram": 20,
        "Vardenafil": 21,
        "Palbociclib": 22,
        "Ricolinostat": 23,
        "Dabrafenib": 24,
        "Proscillaridin A;Proscillaridin-A": 25,
        "IN1451": 26,
        "Ixabepilone": 27,
        "CEP-18770 (Delanzomib)": 28,
        "RG7112": 29,
        "MK-5108": 30,
        "Resminostat": 31,
        "IMD-0354": 32,
        "Alvocidib": 33,
        "LY2090314": 34,
        "Methotrexate": 35,
        "LDN 193189": 36,
        "Tacalcitol": 37,
        "Colchicine": 38,
        "R428": 39,
        "TL_HRAS26": 40,
        "BMS-387032": 41,
        "CGP 60474": 42,
        "TIE2 Kinase Inhibitor": 43,
        "PD-0325901": 44,
        "Isoniazid": 45,
        "GSK-1070916": 46,
        "Masitinib": 47,
        "Saracatinib": 48,
        "CC-401": 49,
        "Decitabine": 50,
        "Ketoconazole": 51,
        "HYDROXYUREA": 52,
        "BAY 61-3606": 53,
        "Navitoclax": 54,
        "Porcn Inhibitor III": 55,
        "GW843682X": 56,
        "Prednisolone": 57,
        "Tamatinib": 58,
        "Tosedostat": 59,
        "GSK256066": 60,
        "MGCD-265": 61,
        "AZD-8330": 62,
        "RN-486": 63,
        "Amiodarone": 64,
        "Belinostat": 65,
        "RVX-208": 66,
        "GO-6976": 67,
        "Scriptaid": 68,
        "HMN-214": 69,
        "SB525334": 70,
        "AVL-292": 71,
        "BMS-777607": 72,
        "AZD4547": 73,
        "Foretinib": 74,
        "Tivozanib": 75,
        "Quizartinib": 76,
        "IKK Inhibitor VII": 77,
        "UNII-BXU45ZH6LI": 78,
        "Chlorpheniramine": 79,
        "Tivantinib": 80,
        "CEP-37440": 81,
        "TPCA-1": 82,
        "AZ628": 83,
        "OSI-930": 84,
        "AZD3514": 85,
        "Vanoxerine": 86,
        "PF-03814735": 87,
        "MLN 2238": 88,
        "Dovitinib": 89,
        "K-02288": 90,
        "Midostaurin": 91,
        "I-BET151": 92,
        "STK219801": 93,
        "PRT-062607": 94,
        "AT 7867": 95,
        "Sunitinib": 96,
        "Penfluridol": 97,
        "BMS-536924": 98,
        "Perhexiline": 99,
        "BI-D1870": 100,
        "FK 866": 101,
        "Mubritinib (TAK 165)": 102,
        "Doxorubicin": 103,
        "Pomalidomide": 104,
        "Colforsin": 105,
        "Phenylbutazone": 106,
        "Protriptyline": 107,
        "Buspirone": 108,
        "Clomipramine": 109,
        "Alogliptin": 110,
        "Nefazodone": 111,
        "ABT737": 112,
        "Dactolisib": 113,
        "Nilotinib": 114,
        "Defactinib": 115,
        "PF-04691502": 116,
        "GLPG0634": 117,
        "Sgc-cbp30": 118,
        "BX 912": 119,
        "SCH-58261": 120,
        "Ruxolitinib": 121,
        "BAY 87-2243": 122,
        "O-Demethylated Adapalene": 123,
        "YK 4-279": 124,
        "Ganetespib (STA-9090)": 125,
        "SLx-2119": 126,
        "Oprozomib (ONX 0912)": 127,
        "Desloratadine": 128,
        "Pitavastatin Calcium": 129,
        "TR-14035": 130,
        "AT13387": 131,
        "CHIR-99021": 132,
        "RG7090": 133,
        "AMD-070 (hydrochloride)": 134,
        "BMS-265246": 135,
        "Tipifarnib": 136,
        "Imatinib": 137,
        "Topotecan": 138,
        "Clemastine": 139,
        "5-(9-Isopropyl-8-methyl-2-morpholino-9H-purin-6-yl)pyrimidin-2-amine": 140,
        "CGM-097": 141,
        "TGX 221": 142,
        "Azacitidine": 143,
        "Atorvastatin": 144,
        "Riociguat": 145,
    }
    return SM_MAP[x]


def create_SM_MAP(train_df):
    sm_names = train_df[["sm_name"]]
    no_duplicates = sm_names.drop_duplicates().reset_index(drop=True)
    SM_MAP = {no_duplicates["sm_name"][i]: i for i in no_duplicates.index}
    return SM_MAP


def use_statistic(df, mapping):
    return np.array([mapping[df.iloc[:, 0][i]] for i in df.index])


class TrainDataset(Dataset):
    def __init__(self, training_df, config, device):
        train_df = training_df
        self.config = config

        main_concat = []
        
        one_hot = train_df.iloc[:,list(range(5+2*18211, 5+2*18211+152))].to_numpy()
        if config["noise"]:
            one_hot = np.concatenate([one_hot, one_hot], axis=0)
        main_concat.append(one_hot)

        if config["chemberta"]:
            chem = train_df.iloc[:,list(range(5+2*18211+152, 5+2*18211+152+600))].to_numpy()
            if config["noise"]:
                chem = np.concatenate([chem, chem], axis=0)
            main_concat.append(chem)
        if config["chemberta_mean"]:
            chem = train_df.iloc[:,list(range(5+2*18211+152+600, train_df.shape[1]))].to_numpy()
            if config["noise"]:
                chem = np.concatenate([chem, chem], axis=0)
            main_concat.append(chem)

        if config["t_score"]:
            train_df = train_df.iloc[:,list(range(0, 5))+list(range(5+18211,5+2*18211))]
        else:
            train_df = train_df.iloc[:,list(range(0, 5+18211))]

        if config["noise"]:
            first = train_df.iloc[:, list(range(0, 5))]
            first = pd.concat([first, first], axis=0).reset_index(drop=True)
            data = train_df.iloc[:, list(range(5, train_df.shape[1]))]
            noise = data.to_numpy()
            for i in range(noise.shape[0]):
                indices = np.random.choice(18211, int(0.3*18211))
                noise[i, indices] = 0
            data = pd.concat([data, pd.DataFrame(noise, columns=data.columns)], axis=0).reset_index(drop=True)
            train_df = pd.concat([first, data], axis=1)
        de_or_t = train_df.iloc[:, list(range(5, train_df.shape[1]))]

        if config["pca"] and config["svd"]:
            raise Exception(
                "Use either PCA or SVD for dimensionality reduction, not both"
            )
        if config["pca"]:
            pca = PCA(n_components=64)
            data = pd.DataFrame(pca.fit_transform(de_or_t.to_numpy())).reset_index(drop=True)
            first = train_df.iloc[:, list(range(0, 5))].reset_index(drop=True)
            train_df = pd.concat([first, data], axis=1)
            self.pca = pca
        if config["svd"]:
            svd = TruncatedSVD(n_components=64)
            data = pd.DataFrame(svd.fit_transform(de_or_t.to_numpy())).reset_index(drop=True)
            first = train_df.iloc[:, list(range(0, 5))].reset_index(drop=True)
            train_df = pd.concat([first, data], axis=1)
            self.svd = svd
        
        cell_df = get_cell_df(train_df)
        sm_df = get_sm_df(train_df)

        if config["emb_cell"]:
            emb_cell = embed_cells(train_df)
            emb_cell = torch.as_tensor(emb_cell.to_numpy()).to(device)
            self.cell_indices = emb_cell

        if config["emb_sm"]:
            emb_sm = embed_sms(train_df)
            emb_sm = torch.as_tensor(emb_sm.to_numpy()).to(device)
            self.sm_indices = emb_sm

        if config["control"]:
            control = is_control(train_df)
            control = control.to_numpy()
            main_concat.append(control)

        if config["mean_cell"]:
            mean_cell = get_cell_means(cell_df)
            self.mean_cell_map = mean_cell
            main_concat.append(use_statistic(cell_df, mean_cell))
        if config["mean_sm"]:
            mean_sm = get_sm_means(sm_df)
            self.mean_sm_map = mean_sm
            main_concat.append(use_statistic(sm_df, mean_sm))
        if config["std_cell"]:
            std_cell = get_cell_stds(cell_df)
            self.std_cell_map = std_cell
            main_concat.append(use_statistic(cell_df, std_cell))
        if config["std_sm"]:
            std_sm = get_sm_stds(sm_df)
            self.std_sm_map = std_sm
            main_concat.append(use_statistic(sm_df, std_sm))
        if config["median_cell"]:
            median_cell = get_cell_medians(cell_df)
            self.median_cell_map = median_cell
            main_concat.append(use_statistic(cell_df, median_cell))
        if config["median_sm"]:
            median_sm = get_sm_medians(sm_df)
            self.median_sm_map = median_sm
            main_concat.append(use_statistic(sm_df, median_sm))

        data = np.concatenate(main_concat, axis=1)
        
        self.dims = data.shape
        self.device = device
        self.X = torch.as_tensor(data, dtype=torch.float32).to(device)

        y = de_or_t.to_numpy()
        if config["inv_transform"]:
            if config["pca"]:
                y = self.pca.transform(y)
            elif config["svd"]:
                y = self.svd.transform(y)
            else:
                raise Exception("No inverse transform without pca or svd")
        self.Y = torch.as_tensor(y, dtype=torch.float32).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.config["emb_cell"] and self.config["emb_sm"]:
            return self.X[idx], self.cell_indices[idx], self.sm_indices[idx], self.Y[idx]

        elif self.config["emb_cell"]:
            return self.X[idx], self.cell_indices[idx], self.Y[idx]

        elif self.config["emb_sm"]:
            return self.X[idx], self.sm_indices[idx], self.Y[idx]

        return self.X[idx], self.Y[idx]


class ValDataset(Dataset):
    """
    Uses statistics computed from the training set to more accurately
    reflect the test set performance
    """
    def __init__(self, validation_df, config, train_dataset):
        val_df = validation_df
        self.config = config

        main_concat = []

        one_hot = val_df.iloc[:,list(range(5+2*18211, 5+2*18211+152))].to_numpy()
        main_concat.append(one_hot)
        
        if config["chemberta"]:
            main_concat.append(val_df.iloc[:,list(range(5+2*18211+152, 5+2*18211+152+600))].to_numpy())
        if config["chemberta_mean"]:
            main_concat.append(val_df.iloc[:,list(range(5+2*18211+152+600, val_df.shape[1]))].to_numpy())

        if config["t_score"]:
            val_df = val_df.iloc[:,list(range(0, 5))+list(range(5+18211,5+2*18211))]
        else:
            val_df = val_df.iloc[:,list(range(0, 5+18211))]

        de_or_t = val_df.iloc[:, list(range(5, val_df.shape[1]))]

        if config["pca"] and config["svd"]:
            raise Exception(
                "Use either PCA or SVD for dimensionality reduction, not both"
            )
        if config["pca"]:
            data = pd.DataFrame(train_dataset.pca.transform(de_or_t))
            first = val_df.iloc[:, list(range(0, 5))].reset_index(drop=True)
            second = data.reset_index(drop=True)
            val_df = pd.concat([first, second], axis=1)
            self.pca = train_dataset.pca
        if config["svd"]:
            data = pd.DataFrame(train_dataset.svd.transform(de_or_t))
            first = val_df.iloc[:, list(range(0, 5))].reset_index(drop=True)
            second = data.reset_index(drop=True)
            val_df = pd.concat([first, second], axis=1)
            self.svd = train_dataset.svd
        
        cell_df = get_cell_df(val_df)
        sm_df = get_sm_df(val_df)

        if config["emb_cell"]:
            emb_cell = embed_cells(val_df)
            self.cell_indices = torch.as_tensor(emb_cell.to_numpy()).to(train_dataset.device)
        if config["emb_sm"]:
            emb_sm = embed_sms(val_df)
            self.sm_indices = torch.as_tensor(emb_sm.to_numpy()).to(train_dataset.device)

        if config["control"]:
            control = is_control(val_df)
            main_concat.append(control.to_numpy())
        
        if config["mean_cell"]:
            main_concat.append(use_statistic(cell_df, train_dataset.mean_cell_map))
        if config["mean_sm"]:
            main_concat.append(use_statistic(sm_df, train_dataset.mean_sm_map))
        if config["std_cell"]:
            main_concat.append(use_statistic(cell_df, train_dataset.std_cell_map))
        if config["std_sm"]:
            main_concat.append(use_statistic(sm_df, train_dataset.std_sm_map))
        if config["median_cell"]:
            main_concat.append(use_statistic(cell_df, train_dataset.median_cell_map))
        if config["median_sm"]:
            main_concat.append(use_statistic(sm_df, train_dataset.median_sm_map))

        data = np.concatenate(main_concat, axis=1)
        
        self.dims = data.shape
        self.X = torch.as_tensor(data, dtype=torch.float32).to(train_dataset.device)
        self.Y = torch.as_tensor(de_or_t.to_numpy(), dtype=torch.float32).to(train_dataset.device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.config["emb_cell"] and self.config["emb_sm"]:
            return self.X[idx], self.cell_indices[idx], self.sm_indices[idx], self.Y[idx]

        elif self.config["emb_cell"]:
            return self.X[idx], self.cell_indices[idx], self.Y[idx]

        elif self.config["emb_sm"]:
            return self.X[idx], self.sm_indices[idx], self.Y[idx]

        return self.X[idx], self.Y[idx]


def calculate_dim(config):
    dim_reduction = config["pca"] or config["svd"]
    dim = 152
    if config["emb_cell"]:
        dim += 64
    if config["emb_sm"]:
        dim += 64
    if config["mean_cell"]:
        dim += (64 if dim_reduction else 18211)
    if config["mean_sm"]:
        dim += (64 if dim_reduction else 18211)
    if config["std_cell"]:
        dim += (64 if dim_reduction else 18211)
    if config["std_sm"]:
        dim += (64 if dim_reduction else 18211)
    if config["median_cell"]:
        dim += (64 if dim_reduction else 18211)
    if config["median_sm"]:
        dim += (64 if dim_reduction else 18211)
    if config["control"]:
        dim += 1
    if config["chemberta"]:
        dim += 600
    if config["chemberta_mean"]:
        dim += 600
    return dim


if __name__ == "__main__":
    save_chemberta_embeddings()