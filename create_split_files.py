# create_split_files.py
import json
import re
import os
from collections import defaultdict


def normalize_whitespace(text):
    """Replaces various whitespace characters with a single space and strips."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_sql(sql_string):
    """Replaces multiple whitespace characters with a single space and strips."""
    # This is primarily needed for comparing SQL strings to find the shortest
    return re.sub(r"\s+", " ", sql_string).strip()


def find_shortest_sql(sql_list):
    """Finds the shortest SQL query, breaking ties alphabetically."""
    if not sql_list:
        return None
    # Sort by length, then alphabetically
    sorted_sql = sorted(sql_list, key=lambda x: (len(x), x))
    return sorted_sql[0]


def substitute_variables_in_string(template_string, variables):
    """Substitutes variable placeholders in a string with their values."""
    substituted_string = template_string
    # Sort keys by length descending to handle cases like city_name1 and city_name
    sorted_vars = sorted(variables.items(), key=lambda item: len(item[0]), reverse=True)
    for var_name, var_value in sorted_vars:
        # Ensure value is treated as string for replacement
        substituted_string = substituted_string.replace(var_name, str(var_value))
    return substituted_string


# We also need a way to substitute variables into SQL templates to get the shortest_sql
# This function is similar to the one in atis_processing, but simplified as we don't
# need var_name_to_type here for this specific output file format.
# We assume variables in template are either "var_name" or var_name.
def substitute_variables_in_sql_for_output(sql_template, variables):
    """Substitutes variable placeholders in an SQL template with their values."""
    substituted_sql = sql_template
    # Sort keys by length descending
    sorted_vars = sorted(variables.items(), key=lambda item: len(item[0]), reverse=True)

    for var_name, var_value in sorted_vars:
        value_str = str(var_value)  # Always convert value to string for replacement

        # Replace quoted versions first
        quoted_placeholder = f'"{var_name}"'
        quoted_value = f'"{value_str}"'  # Assume values in SQL should be quoted strings
        substituted_sql = substituted_sql.replace(quoted_placeholder, quoted_value)

        # Replace unquoted versions
        # This is a heuristic, might need adjustment if templates are complex
        if var_name in substituted_sql:
            substituted_sql = substituted_sql.replace(var_name, value_str)

    return substituted_sql


def create_split_files(atis_json_path, output_base_dir="processed_atis_files"):
    """
    Processes atis.json and creates train.json, dev.json, test.json files
    for both question-split and query-split modes in the specified format.

    Args:
        atis_json_path (str): Path to the original atis.json file.
        output_base_dir (str): Base directory to save the output files.
    """
    print(f"Loading data from {atis_json_path}...")
    try:
        with open(atis_json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {atis_json_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {atis_json_path}.")
        return

    # --- Step 1: Identify Unique Templates and Assign IDs ---
    # This maps the shortest SQL *string* (with placeholders) to a unique integer ID.
    template_to_id = {}
    next_template_id = 0
    print("Identifying unique SQL templates...")
    for query_group in data:
        shortest_sql_template = find_shortest_sql(query_group.get("sql", []))
        if shortest_sql_template is None:
            continue

        if shortest_sql_template not in template_to_id:
            template_to_id[shortest_sql_template] = next_template_id
            next_template_id += 1
    print(f"Found {next_template_id} unique SQL templates.")

    # --- Step 2 & 3: Process Each Sentence and Collect Data for Splits ---
    all_processed_examples_question_split = defaultdict(list)
    all_processed_examples_query_split = defaultdict(list)

    print("Processing sentences and assigning to splits...")
    from tqdm.auto import tqdm  # Import tqdm here if not using it globally

    for query_group in tqdm(data, desc="Processing query groups"):
        group_sql_list = query_group.get("sql", [])
        if not group_sql_list:
            continue

        # Find the shortest SQL template string for this group
        shortest_sql_template = find_shortest_sql(group_sql_list)
        template_id = template_to_id.get(shortest_sql_template)

        group_query_split = query_group.get("query-split", "unknown")

        sentences = query_group.get("sentences", [])
        if not sentences:
            continue

        for sentence_info in sentences:
            input_text_template = sentence_info.get("text", "")
            sentence_variables = sentence_info.get(
                "variables", {}
            )  # Variables specific to this sentence
            sentence_question_split = sentence_info.get("question-split", "unknown")

            # Construct the final input question string by substituting variables
            input_question_with_values = substitute_variables_in_string(
                input_text_template, sentence_variables
            )

            # Construct the substituted shortest SQL query for this specific sentence
            shortest_gold_sql_with_value = substitute_variables_in_sql_for_output(
                shortest_sql_template, sentence_variables
            )
            shortest_gold_sql_with_value = normalize_sql(
                shortest_gold_sql_with_value
            )  # Normalize it

            # --- Create the example dictionary in the requested format ---
            example_data = {
                "question": input_question_with_values,
                "template_id": template_id,  # ID of the shortest template for this group
                "variables": sentence_variables,  # The specific variable map for this sentence
                "shortest_sql": shortest_gold_sql_with_value,  # Added the substituted shortest SQL
                # We don't add the "split" key here; it's added when writing based on which list the example belongs to.
                # Keep original splits for internal tracking/debugging if needed
                "_original_question_split": sentence_question_split,
                "_original_query_split": group_query_split,
            }

            # --- Assign to the correct split lists based on BOTH split types ---
            # Only add to lists if the original split is a valid split name
            if sentence_question_split in ["train", "dev", "test"]:
                all_processed_examples_question_split[sentence_question_split].append(
                    example_data
                )

            if group_query_split in ["train", "dev", "test"]:
                all_processed_examples_query_split[group_query_split].append(
                    example_data
                )

    # --- Step 4: Write Data to JSON Files ---

    # Write files for Question Split
    question_split_output_dir = os.path.join(output_base_dir, "question_split")
    os.makedirs(question_split_output_dir, exist_ok=True)
    print(f"\nWriting question-split files to {question_split_output_dir}...")
    for split_name in ["train", "dev", "test"]:
        file_path = os.path.join(question_split_output_dir, f"{split_name}.json")
        # For the final output files, add the 'split' field to each example dict
        # and filter out internal keys starting with _
        output_examples = []
        for ex in all_processed_examples_question_split[split_name]:
            output_ex = {k: v for k, v in ex.items() if not k.startswith("_")}
            output_ex["split"] = split_name
            output_examples.append(output_ex)

        print(f"  - {split_name}.json: {len(output_examples)} examples")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(output_examples, f, indent=2)

    # Write files for Query Split
    query_split_output_dir = os.path.join(output_base_dir, "query_split")
    os.makedirs(query_split_output_dir, exist_ok=True)
    print(f"\nWriting query-split files to {query_split_output_dir}...")
    for split_name in ["train", "dev", "test"]:
        file_path = os.path.join(query_split_output_dir, f"{split_name}.json")
        # For the final output files, add the 'split' field to each example dict
        # and filter out internal keys starting with _
        output_examples = []
        for ex in all_processed_examples_query_split[split_name]:
            output_ex = {k: v for k, v in ex.items() if not k.startswith("_")}
            output_ex["split"] = split_name
            output_examples.append(output_ex)

        print(f"  - {split_name}.json: {len(output_examples)} examples")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(output_examples, f, indent=2)

    print("\nData splitting and file creation complete.")
    print(f"Output files are in the '{output_base_dir}' directory.")


if __name__ == "__main__":
    # Specify the path to your original atis.json file
    atis_input_file = "atis.json"
    # Specify the base directory where you want to save the split files
    output_directory = "processed_atis_files"  # Using a new directory name

    create_split_files(atis_input_file, output_directory)
