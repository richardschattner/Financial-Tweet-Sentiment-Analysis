# The distilled Model:
# -------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

class EntitySentimentTransformer(nn.Module):
    """The full model.
        Inputs: (batches) of (masked text, mask dictionary)
        Outputs: pooled sentiment logits of entire sentence, dict of sentiment logits per entity
    """
    def __init__(
        self,
        vocab_size,
        max_length=128,
        embed_dim=192,
        num_heads=6,
        num_layers=5,
        ffn_dim= 3 * 192,
        dropout=0.1,
        pad_token_id=None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx = pad_token_id)
        self.position_embeddings = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        # Sentiment head for per-entity scalar
        self.entity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3)
        )

        # Optional pooled head (for distillation)
        self.pooled_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3)
        )

    def forward(self, input_ids, entity_positions=None, attention_mask = None):
        """
        input_ids: [batch, seq_len]
        entity_positions: List[List[int]] of entity token indices per example
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embedding + positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.dropout(x)

        # Transformer encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask = attention_mask)

        x = self.norm(x)

        # Pooled [CLS] embedding (assume first token is [CLS])
        pooled = x[:, 0]

        # Optional pooled head output
        pooled_logits = self.pooled_head(pooled)

        # Per-entity sentiment
        entity_outputs = []
        if entity_positions is not None:
            for i in range(batch_size):
                example_outputs = []
                for idx_list in entity_positions[i]:
                    # idx_list: list of token indices for this entity
                    entity_embed = x[i, idx_list, :]
                    sentiment = self.entity_head(entity_embed)
                    example_outputs.append(sentiment)

                # if no entities found, add one dummy vector so stack() doesn’t fail
                if len(example_outputs) == 0:
                    example_outputs.append(torch.zeros(3, device=device))

                entity_outputs.append(torch.stack(example_outputs))
        else:
            entity_outputs = None

        return {
            "pooled_logits": pooled_logits,   # for distillation
            "entity_logits": entity_outputs    # per-entity sentiment
        }

class TransformerEncoderLayer(nn.Module):
    """Multihead attention encoder block"""
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attention_mask = None):
        # Self-attention
        attn_out, _ = self.self_attn(
                        x, x, x, 
                        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# -------------------------------------------------------------------------------------------
