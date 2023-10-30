import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv, global_add_pool

PAD_TOKEN = 33
MAX_TOKENS = 121

class GIN(torch.nn.Module):
    def __init__(self, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(9, dim_h, bias=False),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU(),
            ),
            edge_dim=3,
        )
        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(dim_h, dim_h, bias=False),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU(),
            ),
            edge_dim=3,
        )
        self.conv3 = GINEConv(
            nn.Sequential(
                nn.Linear(dim_h, dim_h, bias=False),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU(),
            ),
            edge_dim=3,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)

        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        h = torch.cat((h1, h2, h3), dim=1)

        return h


class RotaryEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(RotaryEmbedding, self).__init__()
        self.inv_freq = 1.0 / (
            10000 ** (torch.arange(0, embed_size, 2).float() / embed_size)
        )
        self.seq_len = None
        self.cos = None
        self.sin = None

    def update(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]
        # Reset the tables if the sequence length has changed
        if seq_len != self.seq_len or self.cos.device != x.device:
            self.seq_len = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(
                self.inv_freq
            )
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.cos = emb.cos()[None, None, :, :]
            self.sin = emb.sin()[None, None, :, :]

        return self.cos, self.sin

    def forward(self, q, k):
        self.cos, self.sin = self.update(k, seq_dimension=2)
        cos_q = self.cos[:, :, : q.shape[-2], :]
        sin_q = self.sin[:, :, : q.shape[-2], :]

        cos_k = self.cos[:, :, : k.shape[-2], :]
        sin_k = self.sin[:, :, : k.shape[-2], :]

        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        rotate_q = torch.cat((-q2, q1), dim=-1)
        rotate_k = torch.cat((-k2, k1), dim=-1)

        rotary_embedding_q = (q * cos_q) + (rotate_q * sin_q)
        rotary_embedding_k = (k * cos_k) + (rotate_k * sin_k)

        return rotary_embedding_q, rotary_embedding_k


class RotarySelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(RotarySelfAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != embed_dim:
            raise Exception(
                f"Embed Dim: {self.embed_dim} and Number of Heads: {self.num_heads} are not compatible "
            )

        self.rotary_embedding = RotaryEmbedding(self.head_dim)

        self.qkv_weights = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.final_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask):
        batch_size, seq_length, embed_dim = x.shape
        qkv = self.qkv_weights(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, 3*head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # (batch_size, num_heads, seq_length, head_dim)
        q, k = self.rotary_embedding(q, k)
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / (d_k**0.5)
        attn_logits = attn_logits.masked_fill(mask, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(
            attention, v
        )  # (batch_size, num_heads, seq_length, head_dim)
        values = values.permute(
            0, 2, 1, 3
        )  # (batch_size, seq_length, num_heads, head_dim)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        out = self.final_projection(values)  # (batch_size, seq_length, embed_dim)
        return out


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.attention = RotarySelfAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, output_dim, seq_len=MAX_TOKENS):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.blocks = [Block(embed_dim, num_heads) for _ in range(num_layers)]
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(seq_len*embed_dim, output_dim)

    def forward(self, tokens, mask):
        x = self.embedding(tokens)
        batch_size, seq_len, embed_dim = x.shape
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x).view(batch_size, seq_len*embed_dim)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = OutputModel(example_args_dict)
    total_params = format(
        sum(p.numel() for p in model.parameters() if p.requires_grad), ","
    )
    print(f"Total Number of Parameters: ", total_params)