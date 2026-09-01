"""
Input validation utilities for the ticket clustering tool.
Validates Excel files, settings, and user inputs before processing.
"""

import os


def validate_excel_file(file_path):
    """Validate that a file exists and is a supported Excel format.
    Returns (ok, error_message).
    """
    if not file_path:
        return False, "No file selected."
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in (".xlsx", ".xls"):
        return False, f"Unsupported file type '{ext}'. Use .xlsx or .xls."
    return True, ""


def validate_clustering_settings(num_docs, min_cluster_size, umap_n_neighbors):
    """Validate clustering parameters against the dataset size.
    Returns (ok, error_message).
    """
    errors = []
    if num_docs < 2:
        errors.append("Need at least 2 documents to cluster.")
    if min_cluster_size < 2:
        errors.append("Min cluster size must be at least 2.")
    if min_cluster_size > num_docs:
        errors.append(
            f"Min cluster size ({min_cluster_size}) cannot exceed "
            f"document count ({num_docs})."
        )
    if umap_n_neighbors < 2:
        errors.append("UMAP n_neighbors must be at least 2.")
    if umap_n_neighbors >= num_docs:
        errors.append(
            f"UMAP n_neighbors ({umap_n_neighbors}) must be less than "
            f"document count ({num_docs})."
        )
    if errors:
        return False, "\n".join(errors)
    return True, ""


def validate_model_path(path):
    """Validate that a .gguf model file exists.
    Returns (ok, error_message).
    """
    if not path:
        return False, "No model path specified."
    if not os.path.exists(path):
        return False, f"Model file not found: {path}"
    if not path.lower().endswith(".gguf"):
        return False, "Model file must be a .gguf file."
    return True, ""
