import os
import joblib
from .logging_config import log


def dsave(data, category, name=None, path="result.pkl"):
    if os.path.exists(path):
        try:
            result = joblib.load(path)
        except EOFError:  # Handle corruption
            log.info("Warning: result.pkl is corrupted. Recreating the file...")
            os.remove(path)  # Delete the corrupted file
            result = {"pra": {}, "pra_percomplex": {}, "pr_auc": {}, "tmp": {}, "input": {}}
    else:
        log.progress(f"'{path}' does not exist. Creating a new result structure.")
        result = {"pra": {}, "pra_percomplex": {}, "pr_auc": {}, "tmp": {}, "input": {}}

    if category not in result:
        result[category] = {}

    if name:
        result[category][name] = data
    else:
        result[category] = data

    joblib.dump(result, path)


def dload(category, name=None, path="result.pkl"):
    if os.path.exists(path):
        result = joblib.load(path)
        category_data = result.get(category, {})
        if name:
            return category_data.get(name, {})  # ← return empty dict instead of None
        return category_data
    return {}  # ← fallback if file doesn't exist
