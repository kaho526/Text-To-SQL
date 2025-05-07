# models/ff_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedforwardTemplateClassifier(nn.Module):
    """Feedforward (MLP) model for predicting the SQL template ID."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_templates,
        freeze_embeddings=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_templates)
        self.dropout = nn.Dropout(0.1)

    def forward(self, token_ids, attention_mask=None):
        """
        Args:
            token_ids (torch.Tensor): Input token IDs (batch_size, max_seq_len)
            attention_mask (torch.Tensor, optional): Mask for padded tokens. Defaults to None.
        Returns:
            torch.Tensor: Logits for each template (batch_size, num_templates)
        """
        embedded = self.embedding(token_ids)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            embedded = embedded * mask
            sum_embeddings = torch.sum(embedded, dim=1)
            sum_mask = torch.sum(mask, dim=1)
            avg_embeddings = sum_embeddings / (sum_mask + 1e-9)
        else:
            avg_embeddings = torch.mean(embedded, dim=1)

        hidden = self.fc1(avg_embeddings)
        activated = self.relu(hidden)
        activated = self.dropout(activated)
        logits = self.fc2(activated)

        return logits


class FeedforwardTagger(nn.Module):
    """Feedforward (MLP) model for predicting a tag for each token independently."""

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_tags, freeze_embeddings=False
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_tags)
        self.dropout = nn.Dropout(0.1)

    def forward(self, token_ids):
        """
        Args:
            token_ids (torch.Tensor): Input token IDs (batch_size, max_seq_len)
        Returns:
            torch.Tensor: Logits for each tag per token (batch_size, max_seq_len, num_tags)
        """
        embedded = self.embedding(token_ids)

        hidden = self.fc1(embedded)
        activated = self.relu(hidden)
        activated = self.dropout(activated)
        logits = self.fc2(activated)

        return logits
