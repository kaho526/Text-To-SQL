# models/lstm_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassificationModel(nn.Module):
    """
    A joint LSTM model for ATIS Classification.
    Predicts SQL template ID (from final state) and token tags (from per-step outputs).
    Reference implementation constraints: 1 layer, not bidirectional, no additional FF/MLP layers.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_templates,
        num_tags,
        freeze_embeddings=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.tagging_head = nn.Linear(hidden_dim, num_tags)
        self.template_head = nn.Linear(hidden_dim, num_templates)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size, max_seq_len)
            attention_mask (torch.Tensor, optional): Mask for padded tokens.
        Returns:
            tuple: (template_logits (batch_size, num_templates), tagging_logits (batch_size, max_seq_len, num_tags))
        """
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        # Note: Using pad_packed_sequence/pack_padded_sequence is more efficient with LSTMs
        # when attention_mask is available and sequences have varying lengths.
        # The current simplified code doesn't use packing but relies on ignore_index in loss.
        # Adding packing would improve speed but make the code slightly more complex.
        # Sticking to the simplified version without packing as per the "reference implementation" note.

        output, (h_n, c_n) = self.lstm(embedded)

        output = self.dropout(output)  # Apply dropout after LSTM

        tagging_logits = self.tagging_head(output)

        final_hidden_state = h_n.squeeze(0)  # shape (batch_size, hidden_dim)
        template_logits = self.template_head(final_hidden_state)

        return template_logits, tagging_logits
