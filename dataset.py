# dataset.py
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding  # Requires pip install transformers


class ATISDataset(Dataset):
    """PyTorch Dataset for ATIS data processed by atis_processing.py."""

    def __init__(self, examples):
        # Filter for examples that have Classification data (input_ids, tag_ids, template_id, etc.)
        # This Dataset is specifically for the Classification task where these fields are required.
        self.examples = [ex for ex in examples if ex.get("input_ids") is not None]

        if not self.examples and examples:
            print(
                "Warning: ATISDataset initialized with examples, but none have Classification data (input_ids missing?)."
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        # Return fields required for Classification training/evaluation
        # These fields are guaranteed to exist because of the filtering in __init__
        return {
            "input_ids": item["input_ids"],  # List of int
            "attention_mask": item["attention_mask"],  # List of int
            "template_id": item["template_id"],  # int
            "tag_ids": item["tag_ids"],  # List of int
            # Include other fields needed for reconstruction and evaluation (passed through by collator)
            "tokens": item["tokens"],  # List of str
            "gold_sqls": item["gold_sqls"],  # List of str
            "template_string_with_placeholders": item[
                "template_string_with_placeholders"
            ],  # str
            "input_question": item["input_question"],  # str (for debugging/logging)
            "variable_map": item["variable_map"],  # dict (for debugging/logging)
        }


# --- Data Collator ---
# Make sure you have transformers installed for DataCollatorWithPadding
class ATISDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Pads input_ids, attention_mask, and tag_ids for Classification.
    Passes other fields through as lists within the batch.
    """

    def __call__(self, features):
        # The ATISDataset already filters for examples with Classification data.
        # The features list here should only contain examples with input_ids etc.

        # Use parent DataCollator for input_ids and attention_mask padding
        # Pass a minimal structure to the parent that it expects
        batch = super().__call__(
            [
                {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
                for f in features
            ]
        )

        # Add Classification-specific targets/metadata to the batch
        batch["template_id"] = torch.tensor(
            [f["template_id"] for f in features], dtype=torch.long
        )

        # Pad tag_ids manually
        max_length = batch["input_ids"].shape[1]
        tag_ids_padded = []
        padding_value = -100  # Ignore index for CrossEntropyLoss

        for f in features:
            tags = f["tag_ids"]
            # Truncate if necessary (should already be handled by max_seq_length in processing)
            if len(tags) > max_length:
                tags = tags[:max_length]

            padded_tags = tags + [padding_value] * (max_length - len(tags))
            tag_ids_padded.append(padded_tags)

        batch["tag_ids"] = torch.tensor(tag_ids_padded, dtype=torch.long)

        # Pass through non-tensor fields as lists within the batch
        batch["tokens"] = [f["tokens"] for f in features]
        batch["gold_sqls"] = [f["gold_sqls"] for f in features]
        batch["template_string_with_placeholders"] = [
            f["template_string_with_placeholders"] for f in features
        ]
        batch["input_question"] = [f["input_question"] for f in features]
        batch["variable_map"] = [f["variable_map"] for f in features]

        return batch
