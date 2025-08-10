import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, f1_score, log_loss, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, root_mean_squared_error)
from sklearn.preprocessing import OneHotEncoder
from torchmetrics.functional.classification import multiclass_calibration_error


# ================================================================
# =                                                              =
# =                   Data operations                            =
# =                                                              =
# ================================================================
def compute_all_metrics(args, y_true: np.ndarray, y_pred: np.ndarray, y_hat: np.ndarray):
    metrics = {}
    if args.task == "classification":
        metrics = compute_classification_metrics(args, y_true, y_pred, y_hat)
    elif args.task == "regression":
        metrics = compute_regression_metrics(y_true, y_pred)
    else:
        raise NotImplementedError("args.task must be either 'classification' or 'regression'")

    return metrics


def compute_classification_metrics(args, y_true: np.ndarray, y_pred: np.ndarray, y_hat: np.ndarray):
    metrics = {}

    # === Convert y_hat to probability ===
    if not (np.abs(np.sum(y_hat, axis=1) - 1) < 1e-6).all():
        y_hat = F.softmax(torch.tensor(y_hat), dim=1).numpy()

    # === General metrics for classification ===
    # https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
    # In the binary case, balanced accuracy is equal to the arithmetic mean of sensitivity (true positive rate) and specificity (true negative rate),
    # or the area under the ROC curve with binary predictions (y_pred) rather than scores (y_hat)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["F1_weighted"] = f1_score(y_true, y_pred, average="weighted")
    metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted")
    metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted")

    # === Metrics for predicted probabilities ===
    """OneHotEncoder will follow the order of categories[i] when transforming X[i]
    e.g., Female -> encoding[0]=1; Male -> encoding[1]=1
    >>> enc = OneHotEncoder(handle_unknown='ignore')
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    OneHotEncoder(handle_unknown='ignore')
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
    array([[1., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0.]])
    """
    y_true_onehot = OneHotEncoder(sparse_output=False, categories=[args.class_encoded_list]).fit_transform(
        y_true.reshape(-1, 1)
    )
    metrics["AUROC_weighted"] = roc_auc_score(y_true_onehot, y_hat, average="weighted")
    metrics["ECE"] = multiclass_calibration_error(
        preds=torch.tensor(y_hat),
        target=torch.tensor(y_true).reshape(-1),
        num_classes=args.full_num_classes_processed,
        norm="l1",
    ).item()

    # === Loss ===
    # https://stackoverflow.com/questions/30972029/how-does-the-class-weight-parameter-in-scikit-learn-work
    # y_hat from lit model is already probabilities, thus using nll_loss with log-probabilities, but it is unstable
    # l1 = l2 = l3
    # l1 = F.cross_entropy(y_hat, y, weight=torch.tensor(class_weight, dtype=torch.float32))
    # l2 = F.nll_loss(F.log_softmax(y_hat, dim=1), torch.tensor(y), weight=torch.tensor(class_weight, dtype=torch.float32))
    # l3 = log_loss(y, F.softmax(y_hat, dim=1), normalize=True, sample_weight=sample_weight)
    # For l2, log_softmax must be used, otherwise, log(softmax(x)) is unstable
    metrics["cross_entropy_loss"] = log_loss(
        y_true,
        y_hat,
        sample_weight=[args.train_class2weight[i.item()] for i in y_true],
    )

    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    metrics = {}

    # === General metrics for regression ===
    metrics["rmse"] = root_mean_squared_error(y_true, y_pred)
    metrics["mse"] = mean_squared_error(y_true, y_pred)
    metrics["r2"] = r2_score(y_true, y_pred)

    return metrics


# ================================================================
# =                                                              =
# =                  Feature selection                           =
# =                                                              =
# ================================================================
def compute_sparsity(coef: np.ndarray):
    """compute the number of selected features

    Args:
        coef (np.ndarray): weights of the first layer of (n_classes, n_features)
    """

    gate = (np.linalg.norm(coef, ord=2, axis=0) != 0).astype(int)
    num_selected_features = np.sum(gate)

    return num_selected_features


def gate_bin2dec(gate: np.ndarray):
    gate_binary_str = "".join(str(b) for b in gate)
    gate_dec_str = str(int(gate_binary_str, 2))

    return gate_dec_str
