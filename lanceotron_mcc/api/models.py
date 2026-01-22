import torch
import torch.nn as nn
import torch.optim as optim
import math

try:
    import lightning as L

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

    # Mock LightningModule if not available so the classes can still be defined
    class L:
        class LightningModule(nn.Module):
            pass


def is_ml_available() -> bool:
    """Check if machine learning dependencies (lightning) are available."""
    return LIGHTNING_AVAILABLE


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[: x.size(0)]
        return x


class SignalViT(L.LightningModule):
    def __init__(
        self,
        input_dim: int = 200,
        patch_size: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert input_dim % patch_size == 0, "input_dim must be divisible by patch_size"

        self.patch_size = patch_size
        self.d_model = d_model
        self.lr = lr

        # Linear projection of patches
        self.patch_to_embedding = nn.Linear(patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=(input_dim // patch_size) + 1
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.to_out = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 1), nn.Sigmoid()
        )

        self.criterion = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, input_dim]
        batch_size = x.shape[0]

        # Reshape to patches: [batch_size, num_patches, patch_size]
        x = x.view(batch_size, -1, self.patch_size)

        # Project patches to embeddings: [batch_size, num_patches, d_model]
        x = self.patch_to_embedding(x)

        # Add CLS token: [batch_size, num_patches + 1, d_model]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Switch to [seq_len, batch_size, d_model] for transformer
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)

        # Encode
        x = self.transformer_encoder(x)

        # Take CLS token output: [batch_size, d_model]
        x = x[0]

        # Classify
        return self.to_out(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def load_model(path: str, device: str = "cpu", **kwargs) -> SignalViT:
    """Load a SignalViT model from a checkpoint or state dict."""
    model = SignalViT(**kwargs)
    state_dict = torch.load(path, map_location=device)
    # Handle both full checkpoints and state dicts
    if "state_dict" in state_dict:
        model.load_state_dict(state_dict["state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
