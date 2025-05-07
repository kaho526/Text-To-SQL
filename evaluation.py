# evaluation.py
import re


def normalize_sql(sql_string):
    """
    Normalizes SQL whitespace by replacing any sequence of whitespace
    characters with a single space and stripping leading/trailing space.
    """
    if not isinstance(sql_string, str):
        return ""

    sql_string = re.sub(r"\s+", " ", sql_string).strip()
    return sql_string


def evaluate(predicted_sql, gold_sqls):
    """
    Evaluates a predicted SQL query against a list of valid gold SQL queries.

    Args:
        predicted_sql (str): The predicted SQL query string (fully substituted).
        gold_sqls (list): A list of valid gold SQL query strings for this example (fully substituted).

    Returns:
        bool: True if the normalized predicted SQL matches any normalized gold SQL, False otherwise.
    """
    if (
        predicted_sql is None
        or not isinstance(predicted_sql, str)
        or predicted_sql == "---INVALID_TEMPLATE---"
        or predicted_sql == "---RECONSTRUCTION_ERROR---"
        or not isinstance(gold_sqls, list)
    ):
        return False

    normalized_predicted = normalize_sql(predicted_sql)

    if not normalized_predicted:
        return False

    normalized_golds = [normalize_sql(s) for s in gold_sqls if isinstance(s, str)]

    if normalized_predicted in normalized_golds:
        return True

    return False
