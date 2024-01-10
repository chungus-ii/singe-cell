import torch
import torch.nn as nn
from data_sc import calculate_dim

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

class ResidualBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            activation,
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            activation,
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.block(x) + x


class ResidualBlock_no_BN(nn.Module):
    def __init__(self, dim, activation=nn.ReLU()):
        super(ResidualBlock_no_BN, self).__init__()
        self.block = nn.Sequential(
            activation,
            nn.Linear(dim, dim),
            activation,
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.block(x) + x


class test_MLP(nn.Module):
    def __init__(self, config):
        super(test_MLP, self).__init__()
        self.config = config
        if config["emb_cell"]:
            self.cell_embedding = nn.Embedding(6, 64)
        if config["emb_sm"]:
            self.sm_embedding = nn.Embedding(146, 64)
        dim = calculate_dim(config)
        self.first = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
        )
        if config["inv_transform"]:
            self.projection = nn.Linear(256, 64)
        else:
            self.projection = nn.Linear(256, 18211)
        
    def forward(self, x):
        if self.config["emb_cell"] and self.config["emb_sm"]:
            cell_embedded = torch.squeeze(self.cell_embedding(x[1]))
            sm_embedded = torch.squeeze(self.sm_embedding(x[2]))
            x = torch.cat((x[0], cell_embedded, sm_embedded), dim=1)
        elif self.config["emb_cell"]:
            cell_embedded = torch.squeeze(self.cell_embedding(x[1]))
            x = torch.cat((x[0], cell_embedded), dim=1)
        elif self.config["emb_sm"]:
            sm_embedded = torch.squeeze(self.sm_embedding(x[1]))
            x = torch.cat((x[0], sm_embedded), dim=1)
        else:
            x = x[0]
        x = self.first(x)
        out = self.projection(x)
        return out


class CustomTransformer_v3(nn.Module):
    def __init__(self, config, dropout=0.3):
        super(CustomTransformer_v3, self).__init__()
        self.config = config
        if config["emb_cell"]:
            self.cell_embedding = nn.Embedding(6, 64)
        if config["emb_sm"]:
            self.sm_embedding = nn.Embedding(146, 64)
        dim = calculate_dim(config)
        self.first = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8, dropout=dropout, activation=nn.GELU(), batch_first=True),
            num_layers=6
        )

        if config["inv_transform"]:
            self.projection = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, 64)
            )
        else:
            self.projection = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, 18211)
            )

    def forward(self, x):
        if self.config["emb_cell"] and self.config["emb_cell"]:
            cell_embedded = torch.squeeze(self.cell_embedding(x[1]))
            sm_embedded = torch.squeeze(self.sm_embedding(x[2]))
            x = torch.cat((x[0], cell_embedded, sm_embedded), dim=1)
        elif self.config["emb_cell"]:
            cell_embedded = torch.squeeze(self.cell_embedding(x[1]))
            x = torch.cat((x[0], cell_embedded), dim=1)
        elif self.config["emb_sm"]:
            sm_embedded = torch.squeeze(self.sm_embedding(x[1]))
            x = torch.cat((x[0], sm_embedded), dim=1)
        else:
            x = x[0]

        x = self.first(x)
        x = self.transformer(x)
        out = self.projection(x)

        return out


class ResMLP(nn.Module):
    def __init__(self, config, dropout=0.3):
        super(ResMLP, self).__init__()
        self.config = config
        if config["emb_cell"]:
            self.cell_embedding = nn.Embedding(6, 64)
        if config["emb_sm"]:
            self.sm_embedding = nn.Embedding(146, 64)
        dim = calculate_dim(config)
        self.first = nn.Sequential(
            nn.Linear(dim, 64),
            ResidualBlock_no_BN(64),
            nn.Dropout(dropout),
            ResidualBlock_no_BN(64),
            nn.Dropout(dropout),
            ResidualBlock_no_BN(64),
            nn.Dropout(dropout),
            ResidualBlock_no_BN(64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU()
        )
        if config["inv_transform"]:
            self.projection = nn.Linear(256, 64)
        else:
            self.projection = nn.Linear(256, 18211)

    def forward(self, x):
        if self.config["emb_cell"] and self.config["emb_sm"]:
            cell_embedded = torch.squeeze(self.cell_embedding(x[1]))
            sm_embedded = torch.squeeze(self.sm_embedding(x[2]))
            x = torch.cat((x[0], cell_embedded, sm_embedded), dim=1)
        elif self.config["emb_cell"]:
            cell_embedded = torch.squeeze(self.cell_embedding(x[1]))
            x = torch.cat((x[0], cell_embedded), dim=1)
        elif self.config["emb_sm"]:
            sm_embedded = torch.squeeze(self.sm_embedding(x[1]))
            x = torch.cat((x[0], sm_embedded), dim=1)
        else:
            x = x[0]
        x = self.first(x)
        out = self.projection(x)
        return out


class ResMLP_BN(nn.Module):
    def __init__(self, config, dropout=0.3):
        super(ResMLP_BN, self).__init__()
        self.config = config
        if config["emb_cell"]:
            self.cell_embedding = nn.Embedding(6, 64)
        if config["emb_sm"]:
            self.sm_embedding = nn.Embedding(146, 64)
        dim = calculate_dim(config)
        self.first = nn.Sequential(
            nn.Linear(dim, 64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU()
        )
        if config["inv_transform"]:
            self.projection = nn.Linear(256, 64)
        else:
            self.projection = nn.Linear(256, 18211)

    def forward(self, x):
        if self.config["emb_cell"] and self.config["emb_sm"]:
            cell_embedded = torch.squeeze(self.cell_embedding(x[1]))
            sm_embedded = torch.squeeze(self.sm_embedding(x[2]))
            x = torch.cat((x[0], cell_embedded, sm_embedded), dim=1)
        elif self.config["emb_cell"]:
            cell_embedded = torch.squeeze(self.cell_embedding(x[1]))
            x = torch.cat((x[0], cell_embedded), dim=1)
        elif self.config["emb_sm"]:
            sm_embedded = torch.squeeze(self.sm_embedding(x[1]))
            x = torch.cat((x[0], sm_embedded), dim=1)
        else:
            x = x[0]
        x = self.first(x)
        out = self.projection(x)
        return out


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        dim = calculate_dim(config)
        self.first = nn.Sequential(
            nn.Linear(dim,64),
            ResidualBlock_no_BN(64),
            ResidualBlock_no_BN(64),
            ResidualBlock_no_BN(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.calc_mean = nn.Sequential(
            ResidualBlock_no_BN(64),
            nn.Linear(64, 256)
        )
        self.calc_logvar = nn.Sequential(
            ResidualBlock_no_BN(64),
            nn.Linear(64, 256)
        )
     
    def forward(self, x, cell_embedded, sm_embedded):
        x = torch.cat((x, cell_embedded, sm_embedded), dim=1)
        x = self.first(x)
        mean = self.calc_mean(x)
        logvar = self.calc_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.first = nn.Sequential(
            nn.Linear(256+128, 256),
            ResidualBlock_no_BN(256),
            ResidualBlock_no_BN(256),
        )
        if config["inv_transform"]:
            self.projection = nn.Linear(256, 64)
        else:
            self.projection = nn.Linear(256, 18211)

    def forward(self, z, cell_embedded, sm_embedded):
        z = self.first(torch.cat((z, cell_embedded, sm_embedded), dim=1))
        out = self.projection(z)
        return out


class Encoder_BN(nn.Module):
    def __init__(self, config):
        super(Encoder_BN, self).__init__()
        self.config = config
        dim = calculate_dim(config)
        self.first = nn.Sequential(
            nn.Linear(dim,64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.calc_mean = nn.Sequential(
            ResidualBlock(64),
            nn.Linear(64, 256)
        )
        self.calc_logvar = nn.Sequential(
            ResidualBlock(64),
            nn.Linear(64, 256)
        )
     
    def forward(self, x, cell_embedded, sm_embedded):
        x = torch.cat((x, cell_embedded, sm_embedded), dim=1)
        x = self.first(x)
        mean = self.calc_mean(x)
        logvar = self.calc_logvar(x)
        return mean, logvar


class Decoder_BN(nn.Module):
    def __init__(self, config):
        super(Decoder_BN, self).__init__()
        self.first = nn.Sequential(
            nn.Linear(256+128, 256),
            ResidualBlock(256),
            ResidualBlock(256),
        )
        if config["inv_transform"]:
            self.projection = nn.Linear(256, 64)
        else:
            self.projection = nn.Linear(256, 18211)

    def forward(self, z, cell_embedded, sm_embedded):
        z = self.first(torch.cat((z, cell_embedded, sm_embedded), dim=1))
        out = self.projection(z)
        return out


class cVAE(nn.Module):
    def __init__(self, config, BN):
        super(cVAE, self).__init__()
        self.cell_embedding = nn.Embedding(6, 64)
        self.sm_embedding = nn.Embedding(146, 64)
        if BN:
            self.encoder = Encoder_BN(config)
            self.decoder = Decoder_BN(config)
        else:
            self.encoder = Encoder(config)
            self.decoder = Decoder(config)
        if not(config["emb_cell"] and config["emb_sm"]):
            raise Exception("cVAE must be used with cell and sm embedding tables")

    def sample(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):
        cell_embedded = torch.squeeze(self.cell_embedding(x[1]))
        sm_embedded = torch.squeeze(self.sm_embedding(x[2]))
        mean, logvar = self.encoder(x[0], cell_embedded, sm_embedded)
        z = self.sample(mean, logvar)
        if self.training:
            return self.decoder(z, cell_embedded, sm_embedded), mean, logvar
        else:
            return self.decoder(z, cell_embedded, sm_embedded)