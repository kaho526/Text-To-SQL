import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearTemplateClassifier(nn.Module):
    """Linear model for predicting the SQL template ID (BoW features)."""

    def __init__(self, vocab_size, num_templates):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_templates)
        self.vocab_size = vocab_size  # Store vocab_size for one-hot

    def forward(self, token_ids, attention_mask=None):
        """
        Args:
            token_ids (torch.Tensor): Input token IDs (batch_size, max_seq_len)
            attention_mask (torch.Tensor, optional): Mask for padded tokens.
        Returns:
            torch.Tensor: Logits for each template (batch_size, num_templates)
        """
        # Convert token_ids to one-hot (batch, seq, vocab)
        one_hots = F.one_hot(token_ids, num_classes=self.vocab_size).float()

        # Apply attention mask: zero out one-hots for padded tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (batch_size, max_seq_len, 1)
            one_hots = one_hots * mask

        # Sum one-hots across the sequence dimension (batch, vocab_size)
        sum_one_hots = torch.sum(one_hots, dim=1)

        return self.linear(sum_one_hots)


class LinearTagger(nn.Module):
    """Linear model for predicting a tag for each token independently (One-hot features)."""

    def __init__(self, vocab_size, num_tags):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_tags)
        self.vocab_size = vocab_size  # Store vocab_size

    def forward(self, token_ids):
        """
        Args:
            token_ids (torch.Tensor): Input token IDs (batch_size, max_seq_len)
        Returns:
            torch.Tensor: Logits for each tag per token (batch_size, max_seq_len, num_tags)
        """
        # Convert token_ids to one-hot (batch, seq, vocab)
        one_hots = F.one_hot(token_ids, num_classes=self.vocab_size).float()

        # Apply linear layer per token: (batch, seq, vocab) -> (batch, seq, num_tags)
        logits = self.linear(one_hots)

        return logits
