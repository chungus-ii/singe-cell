import torch
import torch.nn as nn
from torch.nn import functional as F
import math

PAD_TOKEN = 33

#   How we create the node features mask:
#   mask = (node_features.sum(dim=-1) != PAD_TOKEN * embed_size)
#   The mask is in place so that the graph models ignore the padding present in the node features
#   The text models have their own system of padding for the token sequences


example_args_dict = {
    "in_dim": 19,
    "gat_out": 256,
    "gat_heads": 8,
    "gat_dropout": 0.4,
    "vocab_size": 34,
    "text_emb_size": 256,
    "text_nheads": 8,
    "cell_emb_size": 512,
    "concat_nheads": 8,
    "linear_dim": 4096
}

class OutputModel(nn.Module):
    def __init__(self, args_dict):
        super(OutputModel, self).__init__() 
        self.args_dict = args_dict
        self.gat = GATLayer(args_dict["in_dim"], args_dict["gat_out"], args_dict["gat_heads"], args_dict["gat_dropout"])
        self.text_model = RotarySelfAttention(args_dict["vocab_size"], args_dict["text_emb_size"], args_dict["text_nheads"])
        concat_in = args_dict["gat_out"] + args_dict["text_emb_size"] + args_dict["cell_emb_size"]
        self.concat_attention = ConcatAttentionModel(concat_in, args_dict["concat_nheads"], args_dict["cell_emb_size"])
        self.mlp = nn.Sequential(
            nn.Linear(concat_in*2, args_dict["linear_dim"]),
            nn.ReLU(),
            nn.BatchNorm1d(args_dict["linear_dim"]),
            nn.Linear(args_dict["linear_dim"], 18211),
        )

    def forward(self, node_features, adj_mats, tokens, cell_types):
        gat_out = self.gat(node_features, adj_mats)
        text_out = self.text_model(tokens)
        concat_attention = self.concat_attention(gat_out, text_out, cell_types)
        avg_pooled = torch.mean(concat_attention, dim=1)
        max_pooled = torch.max(concat_attention, dim=1)[0]
        concat = torch.cat([avg_pooled, max_pooled], dim=-1)
        output = self.mlp(concat)
        return output



class ConcatAttentionModel(nn.Module):
    def __init__(self, input_size, num_heads, cell_emb_dim):
        super(ConcatAttentionModel, self).__init__()
        self.embedding = nn.Embedding(6, cell_emb_dim, scale_grad_by_freq=True)
        self.cell_emb_dim = cell_emb_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(input_size, num_heads=num_heads)

    def forward(self, gat_output, text_output, cell_type):
        batch_size, num_nodes, _ = gat_output.shape
        cell_type_embedding = self.embedding(cell_type).unsqueeze(1).expand(batch_size, num_nodes, self.cell_emb_dim)
        concatenated_input = torch.cat([gat_output, text_output, cell_type_embedding], dim=-1)
        attention_output, _ = self.attention(concatenated_input.transpose(0, 1), 
                                              concatenated_input.transpose(0, 1),
                                              concatenated_input.transpose(0, 1))
        
        output = attention_output.transpose(0, 1)
        return output


class GATLayer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads=1, dropout=0.5):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise Exception(f"Noncompatible num_heads: {num_heads} and embed_dim: {embed_dim}")

        self.W = nn.Parameter(torch.Tensor(num_heads, input_dim, self.head_dim))
        nn.init.xavier_uniform_(self.W.data, gain=1.38675)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * self.head_dim, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.38675)
        self.linear_out = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A):
        batch_size, num_nodes, _ = X.shape

        # (batch_size, num_heads, num_nodes, head_dim)
        H = torch.einsum("bhni,hid->bhnd", X.unsqueeze(1), self.W)

        # (batch_size, num_heads, num_nodes, 1)
        H1 = torch.einsum("bhnd,hdi->bhni", H, self.a[:, :self.head_dim, :])
        H2 = torch.einsum("bhnd,hdi->bhni", H, self.a[:, self.head_dim:, :])

        # (batch_size, num_heads, num_nodes, num_nodes)
        H1 = H1.expand(-1, self.num_heads, num_nodes, num_nodes)
        H2 = H2.transpose(-1,-2).expand(-1, self.num_heads, num_nodes, num_nodes)
        E = F.leaky_relu(H1 + H2, negative_slope=0.2)

        # (batch_size, num_heads, num_nodes, num_nodes)
        neg_inf = -9e15*torch.ones_like(E)
        A = A.unsqueeze(1)
        attention = torch.where(A > 0, E, neg_inf)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # (batch_size, num_heads, num_nodes, head_dim)
        H_prime = torch.einsum("bhNn,bhnd->bhNd", attention, H)

        # (batch_size, num_heads, embed_dim)
        H_prime = H_prime.reshape(batch_size, num_nodes, self.num_heads * self.head_dim)
        output = self.linear_out(H_prime)
        return output
        


class SelfAttentionTextModel(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length, num_heads, dropout=0.2):
        super(SelfAttentionTextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_TOKEN)
        self.positional_encoding = self.get_positional_encoding(embed_size, max_length)
        self.attention = nn.MultiheadAttention(embed_size, num_heads=num_heads, dropout=dropout)
        self.linear = nn.Linear(embed_size, embed_size)

    def get_positional_encoding(self, embed_size, max_length):
        positional_encoding = torch.zeros(max_length, embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding.unsqueeze(0)

    def forward(self, inputs):
        mask = (inputs != PAD_TOKEN).unsqueeze(1).unsqueeze(2)
        
        embeddings = self.embedding(inputs) + self.positional_encoding
        
        attention_output, _ = self.attention(embeddings.transpose(0, 1), embeddings.transpose(0, 1), embeddings.transpose(0, 1), key_padding_mask=mask)
        output = self.linear(attention_output.transpose(0, 1))
        return output



class RotaryEmbedding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_size, 2).float() / embed_size))
        self.seq_len = None
        self.cos = None
        self.sin = None

    def update(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]
        # Reset the tables if the sequence length has changed
        if seq_len != self.seq_len or self.cos.device != x.device:
            self.seq_len = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
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
    def __init__(self, vocab_size, embed_dim, num_heads, dropout=0.2):
        super(RotarySelfAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim*num_heads) != embed_dim:
            raise Exception(f"Embed Dim: {self.embed_dim} and Number of Heads: {self.num_heads} are not compatible ")

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.rotary_embedding = RotaryEmbedding(self.head_dim)

        self.qkv_weights = nn.Linear(embed_dim, 3*embed_dim, bias = False)
        self.final_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs):
        batch_size, seq_length = inputs.shape
        mask = (inputs == PAD_TOKEN).unsqueeze(2).expand(batch_size, seq_length, seq_length) # (batch_size, seq_length, 1)
        mask = mask + mask.transpose(-1, -2)
        mask = (mask > 0).unsqueeze(1)
        x = self.embedding(inputs)
        qkv = self.qkv_weights(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_length, 3*head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # (batch_size, num_heads, seq_length, head_dim)
        q, k = self.rotary_embedding(q, k)
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attn_logits = attn_logits.masked_fill(mask, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v) # (batch_size, num_heads, seq_length, head_dim)
        values = values.permute(0, 2, 1, 3) # (batch_size, seq_length, num_heads, head_dim)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        out = self.final_projection(values) # (batch_size, seq_length, embed_dim)
        return out

if __name__ == "__main__":
    model = OutputModel(example_args_dict)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Number of Parameters: {total_params}")
