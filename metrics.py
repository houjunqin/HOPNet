from sympy import im


import torch
import torch.nn.functional as F
from typing import Tuple, Dict
from scripts.generate_datasets import CoastalGraph


def continuity_residual(
    eta_t: torch.Tensor,
    eta_tp1: torch.Tensor,
    edge_index: torch.Tensor,
    dt: float = 1.0,
) -> torch.Tensor:
    # Very simple graph divergence proxy for the synthetic benchmark.
    src, dst = edge_index
    B, N = eta_t.shape
    flux = eta_t[:, src] - eta_t[:, dst]
    div = torch.zeros(B, N, device=eta_t.device, dtype=eta_t.dtype)
    div.index_add_(1, dst, flux)
    return (eta_tp1 - eta_t) / dt + 0.1 * div


def boundary_loss(pred: torch.Tensor, target_bc: torch.Tensor, open_boundary_mask: torch.Tensor) -> torch.Tensor:
    mask = open_boundary_mask.unsqueeze(0).unsqueeze(-1)
    return F.l1_loss(pred[mask.expand_as(pred)], target_bc.expand_as(pred)[mask.expand_as(pred)])


def compute_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev_state: torch.Tensor,
    graph: CoastalGraph,
    lambda_phys: float = 0.1,
    lambda_bc: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    data = F.l1_loss(pred, target)
    res = continuity_residual(prev_state[..., 0], pred[..., 0], graph.edge_index, dt=1.0)
    phys = (res ** 2).mean()
    bc = boundary_loss(pred[..., :1], target[..., :1], graph.open_boundary_mask)
    total = data + lambda_phys * phys + lambda_bc * bc
    metrics = {
        "loss": float(total.detach().cpu()),
        "data": float(data.detach().cpu()),
        "phys": float(phys.detach().cpu()),
        "bc": float(bc.detach().cpu()),
        "rmse_eta": float(torch.sqrt(F.mse_loss(pred[..., 0], target[..., 0])).detach().cpu()),
    }
    return total, metrics


import torch
import numpy as np

def masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Mean Absolute Error (MAE) between the predicted and target values,
    while ignoring the entries in the target tensor that match the specified null value.

    This function is particularly useful for scenarios where the dataset contains missing or irrelevant
    values (denoted by `null_val`) that should not contribute to the loss calculation. It effectively
    masks these values to ensure they do not skew the error metrics.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Default is `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean absolute error.

    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = torch.abs(prediction - target)
    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)

def masked_mape(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Mean Absolute Percentage Error (MAPE) between predicted and target values,
    ignoring entries that are either zero or match the specified null value in the target tensor.

    This function is particularly useful for time series or regression tasks where the target values may 
    contain zeros or missing values, which could otherwise distort the error calculation. The function 
    applies a mask to ensure these entries do not affect the resulting MAPE.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Defaults to `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean absolute percentage error.

    Details:
        - The function creates two masks:
          1. `zero_mask`: This mask excludes entries in the `target` tensor that are close to zero, 
             since division by zero or near-zero values would result in extremely large or undefined errors.
          2. `null_mask`: This mask excludes entries in the `target` tensor that match the specified `null_val`. 
             If `null_val` is `np.nan`, the mask will exclude `NaN` values using `torch.isnan`.
        
        - The final mask is the intersection of `zero_mask` and `null_mask`, ensuring that only valid, non-zero,
          and non-null values contribute to the MAPE calculation.
    """

    # mask to exclude zero values in the target
    zero_mask = ~torch.isclose(target, torch.tensor(0.0).to(target.device), atol=5e-5)

    # mask to exclude null values in the target
    if np.isnan(null_val):
        null_mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        null_mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    # combine zero and null masks
    mask = (zero_mask & null_mask).float()

    mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)

    loss = torch.abs((prediction - target) / target)
    loss *= mask
    loss = torch.nan_to_num(loss)

    return torch.mean(loss)

def masked_mse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Mean Squared Error (MSE) between predicted and target values,
    while ignoring the entries in the target tensor that match the specified null value.

    This function is useful for scenarios where the dataset contains missing or irrelevant values 
    (denoted by `null_val`) that should not contribute to the loss calculation. The function applies 
    a mask to these values, ensuring they do not affect the error metric.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Defaults to `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean squared error.

    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to maintain unbiased MSE calculation
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = (prediction - target) ** 2  # Compute squared error
    loss *= mask  # Apply mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)  # Return the mean of the masked loss

def masked_rmse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Root Mean Squared Error (RMSE) between predicted and target values,
    ignoring entries in the target tensor that match the specified null value.

    This function is useful for evaluating model performance on datasets where some target values
    may be missing or irrelevant (denoted by `null_val`). The RMSE provides a measure of the average
    magnitude of errors, accounting only for the valid, non-null entries.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Defaults to `np.nan`. The function will ignore all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked root mean squared error.
    """

    return torch.sqrt(masked_mse(prediction=prediction, target=target, null_val=null_val))

def masked_wape(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Weighted Absolute Percentage Error (WAPE) between predicted and target values,
    ignoring entries in the target tensor that match the specified null value.

    WAPE is a useful metric for measuring the average error relative to the magnitude of the target values,
    making it particularly suitable for comparing errors across datasets or time series with different scales.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Defaults to `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked weighted absolute percentage error.
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    mask = mask.float()
    prediction, target = prediction * mask, target * mask

    prediction = torch.nan_to_num(prediction)
    target = torch.nan_to_num(target)

    loss = torch.sum(torch.abs(prediction - target)) / (torch.sum(torch.abs(target)) + 5e-5)

    return loss