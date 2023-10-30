import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import datetime, os

from data import CompoundEncoderDataset, compound_collate_fn
from model import GIN, TextModel

# Metal Performance Shaders: I'm using a macbook, change this to cuda if you run it yourself
device = torch.device("mps")

lr = 5e-4
batch_size = 146
epochs = 1000
submitting = True

PAD_TOKEN = 33
VOCAB_SIZE = 34
SEQ_LENGTH = 121

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def train_compound_encoder():
    compound_dataset = CompoundEncoderDataset(device=device)
    compound_dataloader = DataLoader(
        compound_dataset,
        batch_size=batch_size,
        collate_fn=compound_collate_fn,
        shuffle=True,
    )
    graph_model = GIN(32).to(device)
    graph_model.apply(init_weights)
    text_model = TextModel(VOCAB_SIZE, 32, 4, 4, 16).to(device)
    text_model.apply(init_weights)
    for epoch in range(epochs):
        for batch_idx, (graphs, tokens) in enumerate(compound_dataloader):
            gin_output = graph_model(graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch)
            mask = (
                (tokens == PAD_TOKEN)
                .unsqueeze(2)
                .expand(batch_size, SEQ_LENGTH, SEQ_LENGTH)
            )
            mask = mask + mask.transpose(-1, -2)
            mask = (mask > 0).unsqueeze(1)
            text_output = text_model(tokens, mask.to(device))
            print(epoch)


def train():
    (
        cell_types,
        compound_adjacency_matrices,
        compound_atom_features,
        smiles_tokens,
        targets,
    ) = getParquetData()

    cell_types_t = torch.Tensor(cell_types).to(device, dtype=torch.int)
    compound_adjacency_matrices_t = torch.Tensor(compound_adjacency_matrices).to(
        device, dtype=torch.float32
    )
    compound_atom_features_t = torch.Tensor(compound_atom_features).to(
        device, dtype=torch.float32
    )
    smiles_tokens_t = torch.Tensor(smiles_tokens).to(device, dtype=torch.int)
    targets_t = torch.Tensor(targets).to(device, dtype=torch.float32)

    dataset = TensorDataset(
        cell_types_t,
        compound_adjacency_matrices_t,
        compound_atom_features_t,
        smiles_tokens_t,
        targets_t,
    )

    # Indices where the cell type is "B cells" or "Myeloid cells"
    indices_bm = np.argwhere(cell_types > 3).squeeze()
    # Indices where the cell type is not "B cells" or "Myeloid cells"
    indices_tnk = np.argwhere(cell_types < 4).squeeze()

    dataset_bm = TensorDataset(*(dataset[indices_bm]))
    dataset_tnk = TensorDataset(*(dataset[indices_tnk]))

    # For leave-one-out cross validation, the "left out" samples will only be B cell and Myeloid cell examples.
    # This is to account for the fact that the test only consists of B cell and Myeloid cell data, and that
    # out of the 614 training examples, only 34 are B cell are Myeloid cell examples.

    model = OutputModel(model_args)
    total_params = format(
        sum(p.numel() for p in model.parameters() if p.requires_grad), ","
    )
    print(f"Model Loaded ({total_params} Parameters)...")
    model = model.to(device)
    model.apply(init_weights)
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    initial_state = model.state_dict()

    print("Starting Training...")
    validation_loss = 0
    for i in range(len(dataset_bm)):
        leftout = dataset_bm[i]
        train_dataset = ConcatDataset(
            [
                TensorDataset(*(dataset_bm[0:i])),
                TensorDataset(*(dataset_bm[i + 1 : len(dataset_bm)])),
                dataset_tnk,
            ]
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        print("-------------------------")
        print(f"Training Fold: {i+1}")
        print("-------------------------")

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_idx, (
                cell_types,
                adj_mats,
                atom_features,
                smiles_tokens,
                targets,
            ) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(atom_features, adj_mats, smiles_tokens, cell_types)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(
                f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss/len(train_loader)}"
            )

        model.eval()
        with torch.no_grad():
            cell_types, adj_mats, atom_features, smiles_tokens, targets = leftout
            outputs = model(
                atom_features.unsqueeze(0),
                adj_mats.unsqueeze(0),
                smiles_tokens.unsqueeze(0),
                cell_types.unsqueeze(0),
            )
            val_loss = criterion(outputs.squeeze(), targets)
        validation_loss += val_loss.item()
        print("-------------------------")
        print(f"Validation Loss: {val_loss.item()}")
        print("-------------------------")
        # model.load_state_dict(initial_state) # keep or not?

    validation_loss /= len(dataset_bm)
    print("=========================")
    print(f"Leave-One-Out Validation Loss: {validation_loss}")
    print("=========================")


if __name__ == "__main__":
    train_compound_encoder()
