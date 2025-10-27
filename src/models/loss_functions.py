"""
Custom loss functions for multi-task learning
Compatible with TensorFlow/Keras backend used by deepctr-torch
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Union, Callable
import numpy as np


def masked_mean_squared_error(y_true, y_pred):
    """
    计算 MSE，但会忽略 y_true 中为 NaN 的样本 (PyTorch版本)
    
    Args:
        y_true: True values tensor
        y_pred: Predicted values tensor
        
    Returns:
        MSE loss ignoring NaN values
    """
    # 1. 创建一个布尔 mask，标记 y_true 中不是 NaN 的位置
    mask = torch.isfinite(y_true)

    # 2. 将 y_true 和 y_pred 中所有 NaN 的位置都变为 0，避免计算错误
    y_true_masked = torch.where(mask, y_true, torch.zeros_like(y_true))
    y_pred_masked = torch.where(mask, y_pred, torch.zeros_like(y_pred))

    # 3. 计算平方误差
    squared_difference = (y_true_masked - y_pred_masked) ** 2

    # 4. 只选取 mask 为 True 的位置的误差
    squared_difference_masked = torch.where(mask, squared_difference, torch.zeros_like(squared_difference))

    # 5. 求平均值时，分母必须是「有效样本的数量」，而不是总样本数
    valid_count = torch.sum(mask.float())
    loss = torch.sum(squared_difference_masked) / torch.clamp(valid_count, min=1.0)

    return loss


def weighted_mse_loss(weight: float):
    """
    创建带权重的MSE损失函数 (PyTorch版本)
    
    Args:
        weight: Task weight multiplier
        
    Returns:
        Weighted MSE loss function
    """
    def loss_fn(y_true, y_pred):
        mse = F.mse_loss(y_pred, y_true, reduction='mean')
        return weight * mse
    
    return loss_fn


def weighted_masked_mse_loss(weight: float):
    """
    创建带权重的Masked MSE损失函数 (PyTorch版本)
    
    Args:
        weight: Task weight multiplier
        
    Returns:
        Weighted masked MSE loss function
    """
    def loss_fn(y_true, y_pred):
        masked_mse = masked_mean_squared_error(y_true, y_pred)
        return weight * masked_mse
    
    return loss_fn


def create_multitask_loss_function(task_names: List[str], 
                                  task_weights: Dict[str, float],
                                  tasks_with_missing: List[str] = None) -> Callable:
    """
    为多任务学习创建单一损失函数 (deepctr-torch兼容)
    
    Args:
        task_names: List of task names in order
        task_weights: Dictionary of task weights
        tasks_with_missing: List of tasks that have missing values (use masked loss)
        
    Returns:
        Single loss function that handles all tasks
    """
    if tasks_with_missing is None:
        tasks_with_missing = []
    
    def multitask_loss(y_pred, y_true, reduction='mean'):
        """
        计算多任务加权损失
        
        Args:
            y_pred: Predictions tensor [batch_size, num_tasks] (deepctr-torch传参顺序)
            y_true: Ground truth tensor [batch_size, num_tasks]
            reduction: Reduction method ('mean', 'sum', 'none')
            
        Returns:
            Weighted sum of task losses
        """
        total_loss = 0.0
        
        for i, task_name in enumerate(task_names):
            weight = task_weights.get(task_name, 1.0)
            
            # 提取当前任务的真实值和预测值
            task_true = y_true[:, i] if y_true.dim() > 1 else y_true
            task_pred = y_pred[:, i] if y_pred.dim() > 1 else y_pred
            
            # 根据是否有缺失值选择损失函数
            if task_name in tasks_with_missing:
                # 对于masked loss，我们需要手动处理reduction
                task_loss = masked_mean_squared_error(task_true, task_pred)
                if reduction == 'sum':
                    # masked_mean_squared_error已经是mean，需要乘以有效样本数
                    valid_count = torch.sum(torch.isfinite(task_true).float())
                    task_loss = task_loss * valid_count
            else:
                task_loss = F.mse_loss(task_pred, task_true, reduction=reduction)
            
            # 添加加权损失
            total_loss += weight * task_loss
        
        return total_loss
    
    return multitask_loss


def create_task_loss_dict(task_names: List[str], 
                         task_weights: Dict[str, float],
                         tasks_with_missing: List[str] = None) -> Dict[str, Union[str, callable]]:
    """
    为多任务学习创建损失函数字典 (TensorFlow/Keras格式)
    
    Args:
        task_names: List of task names
        task_weights: Dictionary of task weights
        tasks_with_missing: List of tasks that have missing values (use masked loss)
        
    Returns:
        Dictionary mapping task names to loss functions
    """
    if tasks_with_missing is None:
        tasks_with_missing = []
    
    loss_dict = {}
    
    for task_name in task_names:
        weight = task_weights.get(task_name, 1.0)
        
        if task_name in tasks_with_missing:
            # Use weighted masked MSE for tasks with missing values
            loss_dict[task_name] = weighted_masked_mse_loss(weight)
        else:
            # Use weighted regular MSE for tasks without missing values
            if weight == 1.0:
                loss_dict[task_name] = 'mse'  # Use string for efficiency
            else:
                loss_dict[task_name] = weighted_mse_loss(weight)
    
    return loss_dict


def calculate_task_weights_from_variance(targets: Dict[str, np.ndarray], 
                                       selected_tasks: List[str],
                                       epsilon: float = 1e-8) -> Dict[str, float]:
    """
    基于方差计算任务权重（方差越小权重越高）
    
    Args:
        targets: Dictionary of target arrays
        selected_tasks: List of selected task names
        epsilon: Small value to avoid division by zero
        
    Returns:
        Dictionary of task weights
    """
    task_variances = {}
    
    for task_name in selected_tasks:
        if task_name in targets:
            variance = np.var(targets[task_name])
            # Add small epsilon to avoid division by zero
            task_variances[task_name] = variance + epsilon
        else:
            task_variances[task_name] = epsilon
            
    # Calculate weights as inverse of variance
    total_inv_var = sum(1.0 / var for var in task_variances.values())
    
    weights = {}
    for task_name, variance in task_variances.items():
        # Normalize weights so they sum to num_tasks
        weights[task_name] = (1.0 / variance) / total_inv_var * len(task_variances)
        
    return weights


def detect_tasks_with_missing_values(targets: Dict[str, np.ndarray], 
                                   selected_tasks: List[str],
                                   threshold: float = 0.01) -> List[str]:
    """
    检测哪些任务包含缺失值
    
    Args:
        targets: Dictionary of target arrays
        selected_tasks: List of selected task names
        threshold: Minimum fraction of missing values to consider a task as having missing data
        
    Returns:
        List of task names with significant missing values
    """
    tasks_with_missing = []
    tasks_missing_ratio = {}
    
    for task_name in selected_tasks:
        if task_name in targets:
            target_array = targets[task_name]
            missing_ratio = np.isnan(target_array).sum() / len(target_array)
            
            if missing_ratio > threshold:
                tasks_with_missing.append(task_name)
                tasks_missing_ratio[task_name] = missing_ratio
    return tasks_with_missing, tasks_missing_ratio