# eval_classification.py
import torch
import json
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import ast

# Assume these files exist
from atis_processing import process_atis_data, reconstruct_sql, normalize_sql
from evaluation import evaluate
from dataset import ATISDataset, ATISDataCollatorWithPadding

# Assume models are in a 'models' subdirectory
from models.linear_classifier import LinearTemplateClassifier, LinearTagger
from models.ff_classifier import FeedforwardTemplateClassifier, FeedforwardTagger
from models.lstm_classifier import LSTMClassificationModel  # <-- Import LSTM


def evaluate_saved_model(
    model_type,
    split_mode,
    data_path="atis.json",
    model_dir="saved_models",
    batch_size=8,
):
    """
    Loads a trained Classification model and evaluates it on the test set.

    Args:
        model_type (str): The type of model to evaluate ('Linear', 'Feedforward', 'LSTM').
        split_mode (str): 'question-split' or 'query-split'. Determines which
                          test set to use and which trained model version to load.
        data_path (str): Path to the original atis.json file.
        model_dir (str): Directory containing saved model runs.
        batch_size (int): Batch size for evaluation DataLoader.
    """
    print(f"--- Evaluating {model_type} model on {split_mode} test split ---")

    # Find the latest saved model directory for this type/split
    model_pattern = f"{model_type.lower()}_{split_mode}_"
    all_saved_runs = [
        d
        for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d)) and d.startswith(model_pattern)
    ]

    if not all_saved_runs:
        print(
            f"Error: No saved models found for {model_type} on {split_mode} split in {model_dir}"
        )
        print("Please train the model first using train_classification.py.")
        return 0.0

    # Sort by name (which includes timestamp) to get the latest one
    latest_run_dir = sorted(all_saved_runs)[-1]
    save_path = os.path.join(model_dir, latest_run_dir)

    metadata_path = os.path.join(save_path, "metadata.json")

    # Determine model weight file names based on model type
    if model_type in ["LSTM"]:  # <-- Check for joint models
        model_weights_path = os.path.join(save_path, "joint_model.pt")
        template_model_path = None  # Not used for joint models
        tagger_model_path = None  # Not used for joint models
    elif model_type in ["Linear", "Feedforward"]:  # <-- Check for separate models
        model_weights_path = None  # Not used for separate models
        template_model_path = os.path.join(save_path, "template_model.pt")
        tagger_model_path = os.path.join(save_path, "tagger_model.pt")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if (
        not os.path.exists(metadata_path)
        or (model_type in ["LSTM"] and not os.path.exists(model_weights_path))
        or (
            model_type in ["Linear", "Feedforward"]
            and (
                not os.path.exists(template_model_path)
                or not os.path.exists(tagger_model_path)
            )
        )
    ):
        print(f"Error: Incomplete model files found in {save_path}")
        print("Please check the saved directory.")
        return 0.0

    print(f"Loading model from: {save_path}")

    # --- Load metadata ---
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    tokenizer_name = metadata.get("tokenizer_name", "bert-base-uncased")
    max_seq_length = metadata.get("max_seq_length", 128)

    # Load mappings and vocab
    template_id_to_string = {
        int(k): v for k, v in metadata.get("template_id_to_string", {}).items()
    }
    template_vars_map_str_keys = metadata.get("template_vars_map", {})
    template_vars_map = {int(k): list(v) for k, v in template_vars_map_str_keys.items()}

    default_values_str_keys = metadata.get("default_values", {})
    # Convert default_values keys back from string tuple representation "(tid, 'var')" to actual tuple (tid, 'var')
    try:
        default_values = {
            ast.literal_eval(k): v for k, v in default_values_str_keys.items()
        }
    except Exception as e:
        print(
            f"Error evaluating default_values keys from metadata: {e}. Defaulting to empty dict (may fail)."
        )
        default_values = {}  # Fallback if parsing fails

    tag_vocab = metadata.get("tag_vocab", ["O"])
    var_name_to_type = metadata.get("var_name_to_type", {})

    # Load model architecture parameters
    num_templates = metadata.get("num_templates", len(template_id_to_string))
    num_tags = metadata.get("num_tags", len(tag_vocab))
    vocab_size = metadata.get("vocab_size", None)
    embedding_dim = metadata.get("embedding_dim")  # FF and LSTM
    hidden_dim = metadata.get("hidden_dim")  # FF and LSTM

    if vocab_size is None or embedding_dim is None or hidden_dim is None:
        if model_type in ["Feedforward", "LSTM"]:
            print(
                f"Error: {model_type} model metadata missing vocab_size, embedding_dim, or hidden_dim. Cannot initialize model. Please retrain."
            )
            return 0.0
        # For Linear, only vocab_size is strictly needed for model init, embedding_dim/hidden_dim can be None

    if model_type == "Linear" and vocab_size is None:
        print(
            "Error: Linear model metadata missing vocab_size. Cannot initialize model."
        )
        return 0.0

    # --- Load Test Data ---
    print("Loading and processing test data...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(
            f"Error loading tokenizer {tokenizer_name}: {e}. Please check network/spelling/library version."
        )
        return 0.0

    # Process data using the *training* split_mode to ensure we get the *correct* test set
    processed_data, _, _, _, _, _ = process_atis_data(
        data_path,
        split_type=split_mode,  # Use the split mode the model was trained on
        tokenizer=tokenizer,  # Tokenizer is required for generating input_ids, tags, etc.
        max_seq_length=max_seq_length,  # Use same max_seq_length as training
    )

    test_examples = processed_data.get("test", [])
    # ATISDataset constructor filters for examples with Classification data
    test_dataset = ATISDataset(test_examples)

    if not test_dataset or len(test_dataset) == 0:
        print(
            f"No test examples found with Classification data for split mode '{split_mode}'. Check data processing."
        )
        return 0.0

    data_collator = ATISDataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator
    )

    # --- Model Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    template_model = None
    tagger_model = None
    joint_model = None

    if model_type == "Feedforward":
        template_model = FeedforwardTemplateClassifier(
            vocab_size, embedding_dim, hidden_dim, num_templates
        )
        tagger_model = FeedforwardTagger(
            vocab_size, embedding_dim, hidden_dim, num_tags
        )
    elif model_type == "Linear":
        template_model = LinearTemplateClassifier(
            vocab_size=vocab_size, num_templates=num_templates
        )
        tagger_model = LinearTagger(vocab_size=vocab_size, num_tags=num_tags)
    elif model_type == "LSTM":  # <-- Initialize the joint LSTM model
        joint_model = LSTMClassificationModel(
            vocab_size, embedding_dim, hidden_dim, num_templates, num_tags
        )
        joint_model.to(device)
        # Load state dict for the joint model
        try:
            joint_model.load_state_dict(
                torch.load(model_weights_path, map_location=device)
            )
        except Exception as e:
            print(
                f"Error loading joint model weights from {model_weights_path}: {e}. Ensure model architecture matches saved weights."
            )
            return 0.0

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load saved weights into separate models for Linear/FF
    if model_type in ["Linear", "Feedforward"]:
        try:
            template_model.load_state_dict(
                torch.load(template_model_path, map_location=device)
            )
            tagger_model.load_state_dict(
                torch.load(tagger_model_path, map_location=device)
            )
            template_model.to(device)
            tagger_model.to(device)
        except Exception as e:
            print(
                f"Error loading separate model weights from {template_model_path} / {tagger_model_path}: {e}. Ensure model architecture matches saved weights."
            )
            return 0.0

    # --- Evaluation Loop ---
    # Set models to evaluation mode
    if model_type in ["Linear", "Feedforward"]:
        template_model.eval()
        tagger_model.eval()
    elif model_type in ["LSTM"]:
        joint_model.eval()

    total_examples = 0
    correct_predictions = 0

    # Output files for predictions and gold answers
    output_predictions_path = os.path.join(
        save_path, f"test_predictions_{split_mode}.txt"
    )
    output_golds_path = os.path.join(save_path, f"test_golds_{split_mode}.txt")
    output_details_path = os.path.join(
        save_path, f"test_details_{split_mode}.txt"
    )  # For manual analysis

    print(f"Evaluating on {len(test_dataset)} test examples...")
    with torch.no_grad(), open(
        output_predictions_path, "w", encoding="utf-8"
    ) as pred_f, open(output_golds_path, "w", encoding="utf-8") as gold_f, open(
        output_details_path, "w", encoding="utf-8"
    ) as details_f:

        progress_bar = tqdm(test_dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            # Move tensor fields to device
            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # --- Forward Pass: Get predictions from models ---
            if model_type in ["Linear", "Feedforward"]:
                template_logits = template_model(
                    batch_on_device["input_ids"], batch_on_device["attention_mask"]
                )
                tagger_logits = tagger_model(
                    batch_on_device["input_ids"]
                )  # Taggers don't use attention mask here

            elif model_type in ["LSTM"]:
                template_logits, tagger_logits = joint_model(
                    batch_on_device["input_ids"], batch_on_device["attention_mask"]
                )

            # --- Get Predicted IDs ---
            predicted_template_ids = torch.argmax(template_logits, dim=-1)
            predicted_tag_ids = torch.argmax(tagger_logits, dim=-1)

            # --- Reconstruct SQL and Evaluate ---
            batch_size = batch_on_device["input_ids"].shape[0]
            for i in range(batch_size):
                total_examples += 1

                pred_template_id = predicted_template_ids[i].item()
                pred_tag_ids_i = predicted_tag_ids[i].tolist()

                seq_len = torch.sum(batch_on_device["attention_mask"][i]).item()
                pred_tag_ids_i = pred_tag_ids_i[:seq_len]

                tokens_i = batch_on_device["tokens"][i][:seq_len]

                gold_sqls_i = batch_on_device["gold_sqls"][i]

                # Reconstruct SQL string from predictions and metadata
                # default_values has tuple keys {(int, str): str} from loading
                # template_vars_map has int keys and list values {int: list}
                reconstructed_sql = reconstruct_sql(
                    pred_template_id,
                    pred_tag_ids_i,
                    tokens_i,
                    template_id_to_string,
                    template_vars_map,
                    default_values,
                    tag_vocab,
                    var_name_to_type,
                )

                # Evaluate the reconstructed SQL
                is_correct = evaluate(reconstructed_sql, gold_sqls_i)
                if is_correct:
                    correct_predictions += 1

                # Write results to files
                pred_f.write(reconstructed_sql + "\n")
                gold_f.write(" ||| ".join(gold_sqls_i) + "\n")
                details_f.write(f"--- Example {total_examples} ---\n")
                details_f.write(f"Input: {batch_on_device['input_question'][i]}\n")
                details_f.write(
                    f"Predicted Template ID: {pred_template_id} (True: {batch_on_device['template_id'][i].item()})\n"
                )
                predicted_tag_names_i = [
                    tag_vocab[t] if 0 <= t < len(tag_vocab) else "O"
                    for t in pred_tag_ids_i
                ]
                details_f.write(f"Tokens ({len(tokens_i)}): {tokens_i}\n")
                details_f.write(
                    f"Predicted Tags ({len(predicted_tag_names_i)}): {predicted_tag_names_i}\n"
                )
                # Access true tag IDs and convert to names, handle padding (-100)
                true_tag_ids_i = batch_on_device["tag_ids"][i][:seq_len].tolist()
                details_f.write(
                    f"True Tags ({len(true_tag_ids_i)}): {[tag_vocab[t] if t != -100 else 'PAD' for t in true_tag_ids_i]}\n"
                )
                details_f.write(f"Predicted SQL: {reconstructed_sql}\n")
                details_f.write(f"Gold SQLs: {gold_sqls_i}\n")
                details_f.write(f"Correct: {is_correct}\n\n")

    accuracy = correct_predictions / total_examples if total_examples > 0 else 0
    print(f"--- Evaluation Results ---")
    print(f"Total Examples: {total_examples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Predictions saved to: {output_predictions_path}")
    print(f"Gold answers saved to: {output_golds_path}")
    print(f"Details saved to: {output_details_path}")

    return accuracy


if __name__ == "__main__":
    # Example Usage:
    # To evaluate the latest saved Feedforward model trained on Question Split:
    evaluate_saved_model(
        model_type="Feedforward", split_mode="question-split"
    )  # <-- Set model_type and split_mode

    # You would typically run this script multiple times to evaluate
    # each trained model configuration:
    # evaluate_saved_model(model_type='Linear', split_mode='question-split')
    # evaluate_saved_model(model_type='LSTM', split_mode='question-split')
    # evaluate_saved_model(model_type='Feedforward', split_mode='query-split')
    # evaluate_saved_model(model_type='Linear', split_mode='query-split')
    # evaluate_saved_model(model_type='LSTM', split_mode='query-split')
