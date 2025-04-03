# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np


# class HuberCustomLoss(nn.Module):
#     def __init__(self, threshold=1.0):
#         """
#         Custom Huber Loss with threshold. This loss transitions from mean squared error
#         to mean absolute error depending on the error magnitude relative to the threshold.
        
#         Args:
#             threshold (float): The point at which the loss transitions from MSE to MAE.
#         """
#         super(HuberCustomLoss, self).__init__()
#         self.threshold = threshold

#     def forward(self, y_true, y_pred, sample_weight=None):
#         """
#         Computes the Huber loss between `y_true` and `y_pred`.
        
#         Args:
#             y_true (torch.Tensor): Ground truth values.
#             y_pred (torch.Tensor): Predicted values.
#             sample_weight (torch.Tensor, optional): Weights for each sample. Defaults to None.
        
#         Returns:
#             torch.Tensor: The computed Huber loss.
#         """
#         # Ensure y_true and y_pred are on the same device
#         y_true = y_true.to(y_pred.device)

#         # Filter out non-finite values (infinite or NaN) in y_true
#         finite_mask = torch.isfinite(y_true)
        
#         # Calculate the error (residual)
#         error = y_pred[finite_mask] - y_true[finite_mask]
        
#         # Compute the squared loss and the linear loss
#         abs_error = torch.abs(error)
#         squared_loss = 0.5 * error**2
#         linear_loss = self.threshold * abs_error - 0.5 * self.threshold**2
        
#         # Determine where the error is "small" or "large"
#         is_small_error = abs_error < self.threshold
        
#         # Compute the final loss (use the squared loss for small errors, linear loss for large errors)
#         loss = torch.where(is_small_error, squared_loss, linear_loss)

#         # If sample weights are provided, apply them
#         if sample_weight is not None:
#             # Broadcast the weights to the correct shape
#             # sample_weight = sample_weight.to(y_pred.device)  # Ensure same device
#             sample_weights = torch.stack([sample_weight for i in range(y_true.size(1))], dim=1).to(y_pred.device)
#             loss = loss * sample_weights[finite_mask]
            
#             # Return the mean loss
#             return loss.sum()

#         # Return the mean loss
#         return loss.mean()



# def r_squared(y_true, y_pred):
#     # Calculate the mean of the true values
#     bool_finite = torch.isfinite(y_true)
#     y_mean = torch.mean(y_true[bool_finite])

#     # Calculate the total sum of squares
#     total_sum_of_squares = torch.sum((y_true[bool_finite] - y_mean)**2)

#     # Calculate the residual sum of squares
#     residual_sum_of_squares = torch.sum((y_true[bool_finite] - y_pred[bool_finite])**2)

#     # Calculate R-squared
#     r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)

#     return torch.mean(r2)
