import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=2, hidden=256, n_heads=4, n_layers=1):
        super().__init__()

        # Project state to hidden dim (d_model)
        self.input_proj = nn.Linear(state_dim, hidden)

        # One transformer encoder layer = contains W_Q, W_K, W_V
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Actor and critic heads
        self.actor = nn.Linear(hidden, action_dim) #2
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        """
        x: shape [batch, state_dim]
        We convert it to [batch, seq=1, hidden] for transformer.
        """
        h = self.input_proj(x)            # [B, H]
        h = h.unsqueeze(1)                # [B, 1, H]
        out = self.transformer(h)         # attention happens here
        z = out[:, 0, :]                  # take the token back to [B, H]

        logits = self.actor(z)
        value = self.critic(z)
        return logits, value
