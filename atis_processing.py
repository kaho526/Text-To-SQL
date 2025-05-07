import json
import re
from collections import defaultdict
import unicodedata
import torch
from transformers import AutoTokenizer  # Requires pip install transformers


def normalize_whitespace(text):
    """Replaces various whitespace characters with a single space and strips."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_sql(sql_string):
    """Normalizes SQL whitespace for comparison."""
    return normalize_whitespace(sql_string)


def find_shortest_sql(sql_list):
    """Finds the shortest SQL query string (with placeholders), breaking ties alphabetically."""
    if not sql_list:
        return None
    valid_sql_strings = [s for s in sql_list if isinstance(s, str)]
    if not valid_sql_strings:
        return None

    sorted_sql = sorted(valid_sql_strings, key=lambda x: (len(x), x))
    return sorted_sql[0]


def substitute_variables_in_string(template_string, variables):
    """Substitutes variable placeholders in a string (like NL text) with their values."""
    if not isinstance(template_string, str):
        return ""

    substituted_string = template_string
    sorted_vars = sorted(variables.items(), key=lambda item: len(item[0]), reverse=True)
    for var_name, var_value in sorted_vars:
        substituted_string = substituted_string.replace(var_name, str(var_value))
    return substituted_string


def substitute_variables_in_sql(sql_template, variables, var_name_to_type=None):
    """
    Substitutes variable placeholders in an SQL string (template or gold) with their values.
    Attempts to handle quoted and unquoted placeholders based on variable name.
    """
    if not isinstance(sql_template, str):
        return ""

    substituted_sql = sql_template
    sorted_vars = sorted(variables.items(), key=lambda item: len(item[0]), reverse=True)

    for var_name, var_value in sorted_vars:
        value_str = str(var_value)

        quoted_placeholder = f'"{var_name}"'
        quoted_value = f'"{value_str}"'
        substituted_sql = substituted_sql.replace(quoted_placeholder, quoted_value)

        if var_name in substituted_sql:
            substituted_sql = substituted_sql.replace(var_name, value_str)

    return substituted_sql


def process_atis_data(
    file_path, split_type="question-split", tokenizer=None, max_seq_length=128
):
    """
    Loads and processes ATIS data from the original atis.json file.
    Prepares data for different tasks based on arguments.

    Args:
        file_path (str): Path to the atis.json file.
        split_type (str): 'question-split' or 'query-split'. Used to assign examples
                          to the 'train', 'dev', 'test' keys in the output dictionary.
        tokenizer: A HuggingFace tokenizer (or compatible). REQUIRED for Classification-
                   specific fields (input_ids, tag_ids, etc.) to be generated.
        max_seq_length (int): Maximum sequence length for tokenizer truncation. Only
                              used if tokenizer is provided.

    Returns:
        dict: A dictionary with keys 'train', 'dev', 'test', each containing
              a list of processed examples (dictionaries). The content of
              example dicts depends on whether a tokenizer was provided.
        dict: A mapping from template ID to the template string (with placeholders).
        dict: A mapping from template ID to the set of required variable names (placeholders) in that template.
        dict: A mapping from (template_id, var_name) tuple to a default value string.
        list: A list of unique tag names ('O' + variable names). Only populated if tokenizer is provided.
        dict: A mapping from variable name (placeholder) to its type string.
    """
    print(f"Loading and processing data from {file_path} using '{split_type}' split...")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return {}, {}, {}, {}, [], {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}.")
        return {}, {}, {}, {}, [], {}

    processed_examples = {"train": [], "dev": [], "test": []}

    template_map = {}
    template_strings = []
    template_vars_map = defaultdict(set)
    template_default_values = defaultdict(dict)

    var_name_to_type = {}

    # --- Tagging vocabulary (only used if tokenizer is provided) ---
    tag_vocab = ["O"]
    tag_to_id = {"O": 0}

    # --- Pass 1: Identify Templates, Vars, Defaults, and Global Var Types ---
    print("Pass 1: Identifying templates, variables, and defaults...")
    # Using tqdm inside the function to show progress if called directly
    from tqdm.auto import tqdm

    for query_group in tqdm(data, desc="Pass 1 Query Groups"):
        for var_info in query_group.get("variables", []):
            var_name = var_info.get("name")
            var_type = var_info.get("type")
            if var_name and var_type:
                var_name_to_type[var_name] = var_type

        shortest_sql_template = find_shortest_sql(query_group.get("sql", []))
        if shortest_sql_template is None:
            continue

        if shortest_sql_template not in template_map:
            template_id = len(template_strings)
            template_map[shortest_sql_template] = template_id
            template_strings.append(shortest_sql_template)
        else:
            template_id = template_map[shortest_sql_template]

        potential_vars_in_template = re.findall(
            r'"([a-zA-Z0-9_]+)"|([a-zA-Z0-9_]+)', shortest_sql_template
        )

        required_vars_in_template = set()
        for quoted_match, unquoted_match in potential_vars_in_template:
            var_name = quoted_match if quoted_match else unquoted_match
            if var_name in var_name_to_type:
                required_vars_in_template.add(var_name)

        template_vars_map[template_id].update(required_vars_in_template)

        for var_info in query_group.get("variables", []):
            var_name = var_info.get("name")
            if var_name and var_name in required_vars_in_template:
                if (template_id, var_name) not in template_default_values:
                    template_default_values[(template_id, var_name)] = str(
                        var_info.get("example", "")
                    )

    # Build final tag vocabulary ('O' + unique variable names found globally)
    tag_vocab += sorted(list(var_name_to_type.keys()))
    tag_to_id = {tag: i for i, tag in enumerate(tag_vocab)}
    print(f"Identified {len(var_name_to_type)} unique variable placeholders.")

    # --- Pass 2: Process Each Sentence and Create Examples ---
    print("Pass 2: Processing sentences and creating examples...")
    for query_group in tqdm(data, desc="Pass 2 Query Groups"):
        group_sql_list = query_group.get("sql", [])
        if not group_sql_list:
            continue

        shortest_sql_template = find_shortest_sql(group_sql_list)
        if shortest_sql_template is None:
            continue
        template_id = template_map.get(shortest_sql_template)

        group_query_split = query_group.get("query-split", "unknown")

        sentences = query_group.get("sentences", [])
        if not sentences:
            continue

        for sentence_info in sentences:
            input_text_template = sentence_info.get("text", "")
            sentence_variables = sentence_info.get("variables", {})
            sentence_question_split = sentence_info.get("question-split", "unknown")

            input_question = substitute_variables_in_string(
                input_text_template, sentence_variables
            )

            gold_sqls = [
                substitute_variables_in_sql(
                    sql_template, sentence_variables, var_name_to_type
                )
                for sql_template in group_sql_list
            ]
            gold_sqls = [normalize_sql(s) for s in gold_sqls]

            shortest_gold_sql_with_value = substitute_variables_in_sql(
                shortest_sql_template, sentence_variables, var_name_to_type
            )
            shortest_gold_sql_with_value = normalize_sql(shortest_gold_sql_with_value)

            split = (
                sentence_info.get(split_type, "unknown")
                if split_type == "question-split"
                else query_group.get(split_type, "unknown")
            )

            if split not in ["train", "dev", "test"]:
                continue

            # --- Prepare base example data dictionary ---
            example_data = {
                "input_question": input_question,
                "gold_sqls": gold_sqls,
                "shortest_gold_sql_with_value": shortest_gold_sql_with_value,
                "_original_question_split": sentence_question_split,
                "_original_query_split": group_query_split,
            }

            # --- Generate Classification-specific fields ONLY if tokenizer is provided ---
            if tokenizer:
                try:
                    encoded_input = tokenizer(
                        input_question,
                        padding="do_not_pad",
                        truncation=True,
                        max_length=max_seq_length,
                        return_offsets_mapping=True,
                        add_special_tokens=True,
                    )
                    input_ids = encoded_input["input_ids"]
                    attention_mask = encoded_input["attention_mask"]
                    offset_mapping = encoded_input["offset_mapping"]
                    tokens = tokenizer.convert_ids_to_tokens(input_ids)

                    value_char_spans_with_vars = []
                    for var_name, var_value in sentence_variables.items():
                        value_str = str(var_value)
                        if not value_str:
                            continue

                        start = 0
                        while True:
                            start = input_question.find(value_str, start)
                            if start == -1:
                                break
                            end = start + len(value_str)
                            value_char_spans_with_vars.append((start, end, var_name))
                            start = end

                    value_char_spans_with_vars.sort(key=lambda x: x[0])

                    current_tags = ["O"] * len(input_ids)

                    for token_idx, (token_char_start, token_char_end) in enumerate(
                        offset_mapping
                    ):
                        if (
                            token_char_start is None
                            or token_char_end is None
                            or token_char_start == token_char_end
                        ):
                            continue

                        assigned_var_name = "O"
                        for (
                            value_char_start,
                            value_char_end,
                            var_name,
                        ) in value_char_spans_with_vars:
                            overlap_start = max(token_char_start, value_char_start)
                            overlap_end = min(token_char_end, value_char_end)
                            overlap_length = max(0, overlap_end - overlap_start)
                            token_len = token_char_end - token_char_start
                            value_len = value_char_end - value_char_start

                            if overlap_length > 0 and (
                                overlap_length / (token_len + 1e-9) >= 0.5
                                or overlap_length / (value_len + 1e-9) >= 0.5
                            ):
                                assigned_var_name = var_name
                                break

                        current_tags[token_idx] = assigned_var_name

                    tag_ids = [
                        tag_to_id.get(tag, tag_to_id["O"]) for tag in current_tags
                    ]

                    if not (
                        len(input_ids)
                        == len(tag_ids)
                        == len(tokens)
                        == len(attention_mask)
                    ):
                        print(
                            f"Warning: Length mismatch after tokenization/tagging for {input_question[:50]}... Skipping example."
                        )
                        continue

                    example_data.update(
                        {
                            "template_id": template_id,
                            "variable_map": sentence_variables,
                            "tokens": tokens,
                            "tag_ids": tag_ids,
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "template_string_with_placeholders": shortest_sql_template,
                        }
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to tokenize or tag sentence: {input_question[:100]}... Error: {e}"
                    )
                    continue

            processed_examples[split].append(example_data)

    print(
        f"Finished processing data for split type '{split_type}'. Collected examples per split:"
    )
    print(f"  Train: {len(processed_examples.get('train', []))}")
    print(f"  Dev:   {len(processed_examples.get('dev', []))}")
    print(f"  Test:  {len(processed_examples.get('test', []))}")

    default_values_str_keys = {str(k): v for k, v in template_default_values.items()}
    template_vars_map_list_values = {k: list(v) for k, v in template_vars_map.items()}

    return (
        processed_examples,
        template_map,
        template_vars_map_list_values,
        default_values_str_keys,
        tag_vocab,
        var_name_to_type,
    )


# --- reconstruct_sql function ---
# This function is specifically for the Classification task's output.
# It takes predictions from a Classification model and reconstructs the SQL.
# It needs the metadata (template_id_to_string, template_vars_map, default_values, tag_vocab, var_name_to_type)
# which are returned by process_atis_data.
# This function should be placed in atis_processing.py as it processes data structures defined here
# and is used externally by eval_classification.py.


def reconstruct_sql(
    predicted_template_id,
    predicted_tag_ids,
    tokens,
    template_id_to_string,
    template_vars_map,
    default_values,
    tag_vocab,
    var_name_to_type,
):
    """
    Reconstructs an SQL query string from predicted template ID and predicted token tags.
    This function is specific to the Classification task's output.

    Args:
        predicted_template_id (int): The predicted ID of the SQL template.
        predicted_tag_ids (list): Predicted tag IDs for the input tokens (list of int).
        tokens (list): The input tokens corresponding to predicted_tag_ids (list of strings).
        template_id_to_string (dict): Mapping from template ID (int) to template string (with placeholders).
        template_vars_map (dict): Mapping from template ID (int) to list of required variable names (placeholders).
                                  Note: loaded from JSON metadata, keys are int, values list.
        default_values (dict): Mapping from (template_id, var_name) tuple (str key from JSON load) to default value string.
                               Note: loaded from JSON metadata, keys are str "(tid, 'var_name')". Needs conversion to tuple keys here.
        tag_vocab (list): List of tag names (strings) corresponding to tag IDs.
        var_name_to_type (dict): Mapping from variable name (placeholder) to its type string.

    Returns:
        str: The reconstructed SQL query string, or "---INVALID_TEMPLATE---" if the template ID is invalid.
             Returns "---RECONSTRUCTION_ERROR---" for other issues.
    """
    # Ensure inputs are valid
    if (
        predicted_template_id is None
        or not isinstance(predicted_template_id, int)
        or predicted_template_id not in template_id_to_string
    ):
        return "---INVALID_TEMPLATE---"
    if (
        not isinstance(predicted_tag_ids, list)
        or not isinstance(tokens, list)
        or len(predicted_tag_ids) != len(tokens)
    ):
        print(
            f"Error: Tag/token list length mismatch in reconstruct_sql. Tags: {len(predicted_tag_ids)}, Tokens: {len(tokens)}"
        )
        return "---RECONSTRUCTION_ERROR---"
    if not tokens:
        return "---RECONSTRUCTION_ERROR---"

    template_string = template_id_to_string[predicted_template_id]

    # template_vars_map is loaded from metadata as {int: list}. We need a set for easy lookup.
    required_vars = set(template_vars_map.get(predicted_template_id, []))

    # Map tag IDs back to tag names
    predicted_tags = [
        tag_vocab[tag_id] if 0 <= tag_id < len(tag_vocab) else "O"
        for tag_id in predicted_tag_ids
    ]

    # Extract variable values from predicted tags
    extracted_values = {}  # {var_name: value_string}
    current_var_name = None
    current_value_tokens = []

    for token_idx, (token, tag) in enumerate(zip(tokens, predicted_tags)):
        if tag != "O" and tag in var_name_to_type:  # Only process valid variable tags
            if current_var_name == tag:
                if token.startswith("##"):
                    current_value_tokens.append(token[2:])
                else:
                    current_value_tokens.append(token)
            else:  # New variable or first token of a variable
                if current_var_name is not None and current_value_tokens:
                    extracted_value = "".join(current_value_tokens)
                    extracted_value = extracted_value.replace(
                        " ##", ""
                    )  # Clean up subword join space
                    if (
                        current_var_name not in extracted_values
                    ):  # Keep the first extracted value for a variable
                        extracted_values[current_var_name] = extracted_value

                current_var_name = tag
                if token.startswith("##"):
                    current_value_tokens = [token[2:]]
                else:
                    current_value_tokens = [token]
        else:  # Tag is 'O' or invalid variable name
            if current_var_name is not None and current_value_tokens:
                extracted_value = "".join(current_value_tokens)
                extracted_value = extracted_value.replace(" ##", "")
                if current_var_name not in extracted_values:
                    extracted_values[current_var_name] = extracted_value
            current_var_name = None
            current_value_tokens = []

    if (
        current_var_name is not None and current_value_tokens
    ):  # Handle last variable if sequence ends mid-variable
        extracted_value = "".join(current_value_tokens)
        extracted_value = extracted_value.replace(" ##", "")
        if current_var_name not in extracted_values:
            extracted_values[current_var_name] = extracted_value

    # Final variable mapping for substitution: extracted values + defaults
    substitution_map = {}
    for var_name in required_vars:
        if var_name in extracted_values:
            substitution_map[var_name] = extracted_values[var_name]
        else:
            # Use default if extraction failed or variable wasn't in input text
            # default_values dict is loaded from metadata with string keys "(tid, 'var_name')"
            # We need to convert the key tuple to string to look it up.
            default_key_str = str((predicted_template_id, var_name))
            substitution_map[var_name] = default_values.get(default_key_str, "")

    # Substitute into the template string. Replace longer placeholder names first.
    substitution_candidates = []
    for var_name in required_vars:
        substitution_candidates.append((var_name, substitution_map.get(var_name, "")))

    substitution_candidates.sort(key=lambda x: len(x[0]), reverse=True)

    final_sql = template_string
    for var_name, value in substitution_candidates:
        quoted_placeholder = f'"{var_name}"'
        quoted_value = f'"{value}"'
        final_sql = final_sql.replace(quoted_placeholder, quoted_value)

        if var_name in final_sql:
            final_sql = final_sql.replace(var_name, value)

    return final_sql


# Example Usage (for testing the functions within this file)
# Note: Running this block requires atis.json and a transformers tokenizer.
if __name__ == "__main__":
    print("Running atis_processing.py test block...")
    try:
        # Test data processing for question split
        # Need a tokenizer to get Classification data
        try:
            tokenizer_for_test = AutoTokenizer.from_pretrained("bert-base-uncased")
        except Exception as e:
            print(
                f"Error loading tokenizer for test block: {e}. Skipping detailed test."
            )
            tokenizer_for_test = None  # Proceed without tokenizer if it fails

        (
            processed_data_qs,
            templates_qs,
            template_vars_qs,
            defaults_qs,
            tags_qs,
            var_types_qs,
        ) = process_atis_data(
            "atis.json",
            split_type="question-split",
            tokenizer=tokenizer_for_test,  # Pass the tokenizer
            max_seq_length=128,
        )

        print("\nSample train example (Question Split):")
        if processed_data_qs["train"]:
            import pprint

            pprint.pprint(processed_data_qs["train"][0])

            # Test reconstruction for the first train example (requires Classification data)
            if tokenizer_for_test and "input_ids" in processed_data_qs["train"][0]:
                example = processed_data_qs["train"][0]
                predicted_template_id = example[
                    "template_id"
                ]  # Use the true template ID
                predicted_tag_ids = example["tag_ids"]  # Use the true tag IDs
                tokens = example["tokens"]  # Use the true tokens

                # defaults_qs has string keys like "(tid, 'var_name')"
                # template_vars_qs has int keys {tid: list}

                reconstructed_sql = reconstruct_sql(
                    predicted_template_id,
                    predicted_tag_ids,
                    tokens,
                    templates_qs,  # template_id_to_string {int: str}
                    template_vars_qs,  # template_vars_map {int: list}
                    defaults_qs,  # default_values {str: str} <-- Pass the dict with string keys
                    tags_qs,  # tag_vocab {list of str}
                    var_types_qs,  # var_name_to_type {str: str}
                )
                print("\nTest Reconstruction (using true values/IDs):")
                print(f"Template ID: {predicted_template_id}")
                print(f"Tokens: {tokens}")
                # Need tag_vocab to convert tag_ids to names for printing
                if tags_qs:
                    print(
                        f"Tags: {[tags_qs[t] if t != -100 else 'PAD' for t in predicted_tag_ids]}"
                    )  # Handle -100 padding if present
                print(f"Reconstructed SQL: {reconstructed_sql}")
                print(f"Gold SQLs: {example['gold_sqls']}")
                print(
                    f"Reconstructed matches Gold: {normalize_sql(reconstructed_sql) in [normalize_sql(s) for s in example['gold_sqls']]}"
                )  # Use normalized comparison

        print("\nSample dev example (Question Split):")
        if processed_data_qs["dev"]:
            import pprint

            pprint.pprint(processed_data_qs["dev"][0])

        print("\nSample test example (Question Split):")
        if processed_data_qs["test"]:
            import pprint

            pprint.pprint(processed_data_qs["test"][0])

        # Test processing without tokenizer (for LLM/Gen data)
        print("\nTest data processing WITHOUT tokenizer (for LLM/Gen)...")
        processed_data_no_tok, _, _, _, _, _ = process_atis_data(
            "atis.json",
            split_type="query-split",  # Example using query split
            tokenizer=None,  # Do NOT pass tokenizer
            max_seq_length=128,  # This param is ignored if tokenizer is None
        )
        print("\nSample train example (Query Split, no tokenizer):")
        if processed_data_no_tok["train"]:
            import pprint

            # Should only have input_question, gold_sqls, shortest_gold_sql_with_value, and original splits
            pprint.pprint(processed_data_no_tok["train"][0])

    except ImportError:
        print(
            "\nSkipping test block: Some required libraries ('transformers', 'tqdm') not found. Please install them."
        )
    except FileNotFoundError:
        print(
            "\nSkipping test block: 'atis.json' not found. Please place it in the same directory."
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred during the test block: {e}")
