from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import os


max_tokens = 121
PAD_TOKEN = 33

def getParquetData(BASE_PATH = "data/de_train.parquet"):
    # Read cell_type, SMILES, and gene (target) data from de_train.parquet
    print("Loading Data...")
    ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), BASE_PATH)
    data = pd.read_parquet(ABSOLUTE_PATH, engine='fastparquet')
    cell_types = np.squeeze(data.loc[:, ['cell_type']].to_numpy())
    smiles = np.squeeze(data.loc[:, ['SMILES']].to_numpy())
    targets = data.iloc[:, 5:18216].to_numpy()
    
    # Collect information from compounds, map cell types to integers, and tokenize SMILES
    print("Collecting Features...")
    type_to_num_vect = np.vectorize(type_to_num)
    cell_types = type_to_num_vect(cell_types)
    compound_adjacency_matrices = np.array(smiles_to_adjacency(smiles))
    compound_atom_features = np.array(smiles_to_atom_features(smiles))
    smiles_tokens = np.array([smiles_to_indices(smiles_string) for smiles_string in smiles])
    return cell_types, compound_adjacency_matrices, compound_atom_features, smiles_tokens, targets


def type_to_num(cell_type):
    match cell_type:
        case "NK cells":
            return 0
        case "T cells CD4+":
            return 1
        case "T cells CD8+":
            return 2
        case "T regulatory cells":
            return 3
        case "B cells":
            return 4
        case "Myeloid cells":
            return 5


def smiles_to_adjacency(smiles_stack, max_atoms=max_tokens):
    adj_list = []

    for smiles in smiles_stack:
        mol = Chem.MolFromSmiles(smiles)
        adjacency = None
        # Get the number of atoms in the molecule
        num_atoms = mol.GetNumAtoms()

        # Optionally, limit the number of atoms (pad if fewer, truncate if more)
        if max_atoms is not None:
            if num_atoms <= max_atoms:
                # Pad with zeros
                adjacency = np.zeros((max_atoms, max_atoms), dtype=np.float32)
            else:
                # Truncate extra atoms
                adjacency = np.zeros((max_atoms, max_atoms), dtype=np.float32)
                mol = Chem.Mol(mol.ToBinary()[:Chem.MolToBinary(mol).rindex(b'\x00' * 4) + 4])

        # Create an empty adjacency matrix
        else:
            adjacency = np.zeros((num_atoms, num_atoms), dtype=np.float32)

        # Iterate through bonds and fill the adjacency matrix
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()  # Get bond type as float (e.g., 1.0 for single bond, 2.0 for double)
            adjacency[start, end] = bond_type
            adjacency[end, start] = bond_type  # Assuming the graph is undirected

        adj_list.append(adjacency)

    return adj_list



def smiles_to_atom_features(smiles_stack, max_atoms=max_tokens):
    features_list = []

    for smiles in smiles_stack:
        mol = Chem.MolFromSmiles(smiles)
        atom_features = []

        # Define mapping dictionaries for non-numeric features
        HYBRIDIZATION_MAP = {
            Chem.rdchem.HybridizationType.SP: 0,
            Chem.rdchem.HybridizationType.SP2: 1,
            Chem.rdchem.HybridizationType.SP3: 2,
            Chem.rdchem.HybridizationType.SP3D: 3,
            Chem.rdchem.HybridizationType.SP3D2: 4,
        }

        CHIRALITY_MAP = {
            Chem.ChiralType.CHI_UNSPECIFIED: 0,
            Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
        }

        BOND_TYPE_MAP = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3,
        }

        def bond_type_to_one_hot(bond_types):
            num_bond_types = len(BOND_TYPE_MAP)
            one_hot_encoding = [0] * num_bond_types
            for bond_type in bond_types:
                if bond_type in BOND_TYPE_MAP:
                    one_hot_encoding[BOND_TYPE_MAP[bond_type]] = 1
                else:
                    raise Exception(f"bond type: {bond_type} not accounted for in mapping")
            return one_hot_encoding
        
        # Iterate through atoms and extract atom information
        for atom in mol.GetAtoms():
            # Extract atom features as a list
            atomic_number = atom.GetAtomicNum()
            atomic_mass = atom.GetMass()
            valence_electrons = atom.GetTotalValence()
            implicit_valence = atom.GetImplicitValence()
            hydrogens = atom.GetTotalNumHs()
            hybridization = HYBRIDIZATION_MAP.get(atom.GetHybridization(), -1)  # Map to numerical value
            chirality = CHIRALITY_MAP.get(atom.GetChiralTag(), -1)  # Map to numerical value
            formal_charge = atom.GetFormalCharge()
            neighbors = len(atom.GetNeighbors())
            
            # Extract bond types and convert them to one-hot encoding
            bond_types = [bond.GetBondType() for bond in atom.GetBonds()]
            bond_type_encoding = bond_type_to_one_hot(bond_types)
            
            functional_groups = atom.GetNumRadicalElectrons()
            ring_membership = [int(atom.IsInRing()), len(Chem.GetSymmSSSR(mol))]
            aromaticity = int(atom.GetIsAromatic())
            degree = atom.GetDegree()
            clustering_coefficient = AllChem.CalcCrippenDescriptors(mol)[0]
            
            # Append atom features as a list
            atom_feature = [atomic_number, atomic_mass, valence_electrons, implicit_valence, hydrogens, hybridization, chirality, 
                            formal_charge, neighbors] + bond_type_encoding + [functional_groups] + ring_membership + [aromaticity,
                            degree, clustering_coefficient]
            atom_features.append(atom_feature)
        
        # Pad or truncate features matrix based on max_atoms
        if max_atoms is not None:
            num_atoms = len(atom_features)
            if num_atoms < max_atoms:
                # Pad with zeros
                pad_size = max_atoms - num_atoms
                padding = [[PAD_TOKEN] * len(atom_features[0])] * pad_size
                atom_features.extend(padding)
            elif num_atoms > max_atoms:
                # Truncate extra atoms
                atom_features = atom_features[:max_atoms]
        
        features_list.append(atom_features)

    return features_list


def smiles_to_indices(smiles_string):
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
        "7": 32
    }
    tokens = []
    i = 0
    while i < len(smiles_string):
        # Check for two character tokens (Br, Cl, @@)
        if smiles_string[i:i + 2] in CUSTOM_TOKENS:
            tokens.append(CUSTOM_TOKENS[smiles_string[i:i + 2]])
            i += 2
        # Check for single-character tokens
        elif smiles_string[i] in CUSTOM_TOKENS:
            tokens.append(CUSTOM_TOKENS[smiles_string[i]])
            i += 1
        else:
            raise ValueError(f"Invalid atom: {smiles_string[i]}")
    tokens += [PAD_TOKEN] * (max_tokens - len(tokens))
    return tokens


if __name__ == "__main__":
    cell_types, compound_adjacency_matrices, compound_atom_features, smiles_tokens, targets = getParquetData()
    print(compound_atom_features.shape)