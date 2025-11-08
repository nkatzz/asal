import os
import sys
import time
import random
import numpy as np

sys.path.insert(0, os.path.normpath(os.getcwd() + os.sep + os.pardir))

from src.asal_nesy.dsfa_old.models import DigitCNN
from src.asal_nesy.neurasal.utils import *
from src.asal_nesy.neurasal.data_structs import get_data, get_data_loader, SequenceDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from collections import Counter
import math


# ---------------------------------------------------------
# Positional encoding (sinusoidal, like in "Attention is All You Need")
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / max_len)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 0,2,4,...
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 1,3,5,...
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (bs, seqlen, d_model)
        """
        seqlen = x.size(1)
        x = x + self.pe[:, :seqlen, :]
        return self.dropout(x)


# ---------------------------------------------------------
# CNN + Transformer for sequence classification
# ---------------------------------------------------------
class CNN_Transformer(nn.Module):
    def __init__(
        self,
        cnn_model,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        output_size=1,
        aggregation="concat",
        use_logits=True,
    ):
        """
        aggregation: 'concat' or 'mean'
        use_logits: if True, feed CNN logits to the Transformer; otherwise, features
        """
        super().__init__()
        self.cnn = cnn_model
        self.aggregation = aggregation
        self.use_logits = use_logits

        # We don't know input_size (dim * out_features or cnn_feature_size) until forward, so we lazy-init.
        self.input_size = None
        self.proj = None  # linear projection to d_model (lazy-init)

        # Transformer stack (lazy-init is possible too, but easier to fix d_model and adapt with self.proj)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.posenc = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, dim, channels, height, width)
        """
        bs, seqlen, dim, c, h, w = x.shape
        x = x.view(bs * seqlen * dim, c, h, w)  # (bs*seqlen*dim, c, h, w)

        if self.use_logits:
            # feed logits (your note: works better for you)
            cnn_out = self.cnn(x, apply_softmax=False, return_features=False)
        else:
            # feed features
            cnn_out = self.cnn(x, apply_softmax=False, return_features=True)

        # (bs, seqlen, dim, feat)
        cnn_out = cnn_out.view(bs, seqlen, dim, -1)

        if self.aggregation == "concat":
            cnn_out = cnn_out.view(bs, seqlen, -1)  # (bs, seqlen, dim*feat)
        elif self.aggregation == "mean":
            cnn_out = cnn_out.mean(dim=2)  # (bs, seqlen, feat)
        else:
            raise ValueError("aggregation must be 'concat' or 'mean'")

        # Lazy init projection if needed
        if self.proj is None:
            self.input_size = cnn_out.size(-1)
            self.proj = nn.Linear(self.input_size, self.transformer.layers[0].linear1.in_features).to(cnn_out.device)

        # Project to d_model for the transformer
        x_tok = self.proj(cnn_out)  # (bs, seqlen, d_model)

        # Positional encoding + transformer
        x_tok = self.posenc(x_tok)
        x_tok = self.transformer(x_tok)  # (bs, seqlen, d_model)

        # Pool over time (mean)
        pooled = x_tok.mean(dim=1)  # (bs, d_model)

        logits = self.fc(pooled)  # (bs, 1)
        return torch.sigmoid(logits)  # keep BCELoss compatibility


# ---------------------------------------------------------
# Train / test helpers
# ---------------------------------------------------------
def evaluate(model, data_loader, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        actual, predicted = [], []
        for batch in data_loader:
            tensors = torch.stack([seq.images for seq in batch]).to(device)
            labels = torch.tensor([seq.seq_label for seq in batch]).to(device)

            outputs = model(tensors)
            preds = (outputs >= threshold).long().view(-1)

            actual.extend(labels.cpu().numpy())
            predicted.extend(preds.cpu().numpy())

    return f1_score(actual, predicted, average="binary", zero_division=0)


# ---------------------------------------------------------
if __name__ == "__main__":

    # ------------ reproducibility ------------
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # If you want full determinism on GPU (may slow you down / raise errors for some ops):
    # import os
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------ config ------------
    batch_size = 50
    aggregation = "concat"    # 'concat' or 'mean'
    use_logits = True         # feed CNN logits to the transformer (what you said works better)
    lr = 0.0001
    num_epochs = 2000

    # ------------ data ------------
    train_data, test_data = get_data(
        "/home/nkatz/dev/asal_data/mnist_nesy/len_10_dim_1_pattern_sfa_1/mnist_train.pt",
        "/home/nkatz/dev/asal_data/mnist_nesy/len_50_dim_1_pattern_sfa_1/mnist_test.pt"
    )

    print("Label distribution in train data:", Counter(seq.seq_label for seq in train_data))
    print("Label distribution in test data:", Counter(seq.seq_label for seq in test_data))

    train_loader: DataLoader[SequenceDataset] = get_data_loader(train_data, batch_size=batch_size, train=True)
    test_loader: DataLoader[SequenceDataset] = get_data_loader(test_data, batch_size=batch_size, train=False)

    # ------------ model ------------
    cnn_model = DigitCNN(out_features=10)
    model = CNN_Transformer(
        cnn_model=cnn_model,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        output_size=1,
        aggregation=aggregation,
        use_logits=use_logits,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    print("Training CNN + Transformer baseline...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            tensors = torch.stack([seq.images for seq in batch]).to(device)
            labels = torch.tensor([seq.seq_label for seq in batch]).to(device)

            outputs = model(tensors)
            loss = criterion(outputs.view(-1), labels.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        train_f1 = evaluate(model, train_loader, device)
        test_f1 = evaluate(model, test_loader, device, threshold=0.4)

        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {total_loss / len(train_loader):.3f}, "
            f"Train F1: {train_f1:.3f}, Test F1: {test_f1:.3f}, "
            f"Time: {time.time() - start_time:.3f} secs"
        )
