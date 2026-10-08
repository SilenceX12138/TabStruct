from tabeval.plugins.core.models.tabular_encoder import TabularEncoder


def parse_col_type_info(
    task: str,
    feature_scaler_model: TabularEncoder,
    target_scaler_model: TabularEncoder,
    all_col_cardinality_list: list,
):
    """Parse column type information from the scaler models.

    Args:
        task (str): task type, e.g., "unsupervision", "classification", "regression"
        feature_scaler_model (TabularEncoder): scaler model for features
        target_scaler_model (TabularEncoder): scaler model for target
        all_col_cardinality_list (list): List of cardinalities for each categorical column

    Returns:
        dict: A dictionary containing:
            - order_numerical_list (list): Order of numerical column indices WITHOUT processing
            - order_categorical_list (list): Order of categorical column indices WITHOUT processing
            - indices_numerical_list (list): List of indices for numerical columns
            - indices_categorical_list (list): List of indices for categorical columns
            - full_cardinality_list (list): List of cardinalities for each categorical column in full set
            - train_cardinality_list (list): List of cardinalities for each categorical feature as observed in the training data
    """
    # === Parse column type information from the scaler models ===
    order_numerical_list = []
    order_categorical_list = []
    indices_numerical_list = []
    indices_categorical_list = []
    full_cardinality_list = []
    train_cardinality_list = []
    scaler_list = [feature_scaler_model]
    if task != "unsupervision":
        scaler_list.append(target_scaler_model)

    # === Iterate through both feature and target scalers to gather column info ===
    col_start = 0
    col_idx = 0
    for scaler in scaler_list:
        for col_info in scaler.layout():
            # Each `col_info` contains:
            # - transform: the feature encoder used
            # - feature_type: "continuous" or "discrete"
            # - output_dimensions: int (for continuous, can be >1 for embeddings)
            col_type = col_info.feature_type
            col_length = col_info.output_dimensions
            col_end = col_start + col_length

            # Collect the column indices based on their type
            if col_type == "continuous":
                order_numerical_list.append(col_idx)
                # Note: numerical features can be multi-dimensional (e.g., embeddings)
                indices_numerical_list.extend(list(range(col_start, col_end)))
            elif col_type == "discrete":
                order_categorical_list.append(col_idx)
                # `range` does **not** include the end index
                indices_categorical_list.extend(list(range(col_start, col_end)))
                full_cardinality_list.append(all_col_cardinality_list[col_idx])
                col_encoder = col_info.transform.encoder
                train_cardinality_list.append(len(getattr(col_encoder, "categories_", [[None]])[0]))

            col_start = col_end
            col_idx += 1

    return {
        "order_numerical_list": order_numerical_list,
        "order_categorical_list": order_categorical_list,
        "indices_numerical_list": indices_numerical_list,
        "indices_categorical_list": indices_categorical_list,
        "full_cardinality_list": full_cardinality_list,
        "train_cardinality_list": train_cardinality_list,
    }
