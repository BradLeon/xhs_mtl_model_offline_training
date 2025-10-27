"""
Preprocessing utilities for multi-task learning
Includes label normalization and data quality checks
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class MultiTaskLabelNormalizer:
    """
    Multi-task label normalization utility
    Handles normalization and denormalization for multiple tasks
    """
    
    def __init__(self, normalization_method: str = 'standard'):
        """
        Initialize the normalizer
        
        Args:
            normalization_method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
        """
        self.normalization_method = normalization_method
        self.normalizers = {}
        self.fitted_tasks = set()
        
    def fit_transform(self, targets: Dict[str, np.ndarray], 
                     task_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Fit normalizers on training data and transform
        
        Args:
            targets: Dictionary of target arrays
            task_names: List of task names to normalize
            
        Returns:
            Dictionary of normalized target arrays
        """
        normalized_targets = {}
        
        for task_name in task_names:
            if task_name in targets:
                target_array = targets[task_name].reshape(-1, 1)
                
                # Create normalizer for this task
                if self.normalization_method == 'standard':
                    self.normalizers[task_name] = StandardScaler()
                elif self.normalization_method == 'minmax':
                    self.normalizers[task_name] = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown normalization method: {self.normalization_method}")
                
                # Fit and transform
                normalized = self.normalizers[task_name].fit_transform(target_array)
                normalized_targets[task_name] = normalized.flatten()
                self.fitted_tasks.add(task_name)
                
                # Log normalization statistics
                if self.normalization_method == 'standard':
                    mean = self.normalizers[task_name].mean_[0]
                    std = self.normalizers[task_name].scale_[0]
                    logger.info(f"Normalized {task_name}: mean={mean:.6f}, std={std:.6f}")
                else:
                    min_val = self.normalizers[task_name].data_min_[0]
                    max_val = self.normalizers[task_name].data_max_[0]
                    logger.info(f"Normalized {task_name}: min={min_val:.6f}, max={max_val:.6f}")
            else:
                logger.warning(f"Task {task_name} not found in targets")
                normalized_targets[task_name] = np.zeros(len(list(targets.values())[0]))
                
        return normalized_targets
    
    def transform(self, targets: Dict[str, np.ndarray], 
                 task_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Transform validation/test data using fitted normalizers
        
        Args:
            targets: Dictionary of target arrays
            task_names: List of task names to normalize
            
        Returns:
            Dictionary of normalized target arrays
        """
        normalized_targets = {}
        
        for task_name in task_names:
            if task_name in targets and task_name in self.fitted_tasks:
                target_array = targets[task_name].reshape(-1, 1)
                normalized = self.normalizers[task_name].transform(target_array)
                normalized_targets[task_name] = normalized.flatten()
            elif task_name in targets:
                # No normalizer fitted for this task
                logger.warning(f"No normalizer fitted for {task_name}, using raw values")
                normalized_targets[task_name] = targets[task_name]
            else:
                logger.warning(f"Task {task_name} not found in targets")
                normalized_targets[task_name] = np.zeros(len(list(targets.values())[0]))
                
        return normalized_targets
    
    def inverse_transform(self, predictions: np.ndarray, 
                         task_names: List[str]) -> np.ndarray:
        """
        Denormalize predictions back to original scale
        
        Args:
            predictions: Array of predictions [samples, num_tasks]
            task_names: List of task names corresponding to prediction columns
            
        Returns:
            Denormalized predictions
        """
        denormalized = np.zeros_like(predictions)
        
        for i, task_name in enumerate(task_names):
            if task_name in self.fitted_tasks:
                task_pred = predictions[:, i:i+1]
                denormalized[:, i:i+1] = self.normalizers[task_name].inverse_transform(task_pred)
            else:
                denormalized[:, i] = predictions[:, i]
                
        return denormalized
    
    def get_normalization_stats(self, task_name: str) -> Optional[Dict[str, float]]:
        """
        Get normalization statistics for a task
        
        Args:
            task_name: Name of the task
            
        Returns:
            Dictionary with normalization statistics
        """
        if task_name not in self.fitted_tasks:
            return None
            
        normalizer = self.normalizers[task_name]
        
        if self.normalization_method == 'standard':
            return {
                'mean': float(normalizer.mean_[0]),
                'std': float(normalizer.scale_[0]),
                'var': float(normalizer.var_[0])
            }
        elif self.normalization_method == 'minmax':
            return {
                'min': float(normalizer.data_min_[0]),
                'max': float(normalizer.data_max_[0]),
                'range': float(normalizer.data_range_[0])
            }


def analyze_target_quality(targets: Dict[str, np.ndarray], 
                          task_names: List[str],
                          prefix: str = "") -> Dict[str, Dict[str, Union[float, int]]]:
    """
    分析目标变量的数据质量
    
    Args:
        targets: Dictionary of target arrays
        task_names: List of task names to analyze
        prefix: Prefix for logging messages
        
    Returns:
        Dictionary with quality statistics for each task
    """
    quality_stats = {}
    
    logger.info("="*50)
    logger.info(f"TARGET DATA QUALITY ANALYSIS {prefix}")
    logger.info("="*50)
    
    for task_name in task_names:
        if task_name in targets:
            target_array = targets[task_name]
            
            # Basic statistics
            stats = {
                'count': len(target_array),
                'mean': float(np.mean(target_array)),
                'std': float(np.std(target_array)),
                'min': float(np.min(target_array)),
                'max': float(np.max(target_array)),
                'unique_values': len(np.unique(target_array)),
                'missing_count': int(np.isnan(target_array).sum()),
                'missing_ratio': float(np.isnan(target_array).sum() / len(target_array)),
                'zero_count': int(np.sum(target_array == 0)),
                'zero_ratio': float(np.sum(target_array == 0) / len(target_array))
            }
            
            quality_stats[task_name] = stats
            
            # Log analysis
            logger.info(f"{task_name}:")
            logger.info(f"  Count: {stats['count']}, Unique: {stats['unique_values']}")
            logger.info(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            logger.info(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            logger.info(f"  Missing: {stats['missing_count']} ({stats['missing_ratio']:.2%})")
            logger.info(f"  Zeros: {stats['zero_count']} ({stats['zero_ratio']:.2%})")
            
            # Quality warnings
            if stats['std'] < 1e-8:
                logger.warning(f"  ⚠️  CONSTANT VALUES detected for {task_name}!")
            if stats['missing_ratio'] > 0.1:
                logger.warning(f"  ⚠️  HIGH MISSING RATIO ({stats['missing_ratio']:.2%}) for {task_name}")
            if stats['unique_values'] < 10:
                logger.warning(f"  ⚠️  LOW DIVERSITY ({stats['unique_values']} unique) for {task_name}")
        else:
            logger.warning(f"Task {task_name} not found in targets")
            quality_stats[task_name] = {'error': 'Task not found'}
    
    logger.info("="*50)
    
    return quality_stats


def check_prediction_quality(predictions: np.ndarray, 
                           task_names: List[str],
                           prefix: str = "") -> Dict[str, Dict[str, Union[float, int]]]:
    """
    分析预测值的质量
    
    Args:
        predictions: Array of predictions [samples, num_tasks]
        task_names: List of task names corresponding to prediction columns
        prefix: Prefix for logging messages
        
    Returns:
        Dictionary with prediction quality statistics
    """
    quality_stats = {}
    
    logger.info("="*60)
    logger.info(f"PREDICTION QUALITY ANALYSIS {prefix}")
    logger.info("="*60)
    
    for i, task_name in enumerate(task_names):
        task_predictions = predictions[:, i] if predictions.ndim > 1 else predictions
        
        # Basic statistics
        stats = {
            'count': len(task_predictions),
            'mean': float(np.mean(task_predictions)),
            'std': float(np.std(task_predictions)),
            'min': float(np.min(task_predictions)),
            'max': float(np.max(task_predictions)),
            'unique_values': len(np.unique(task_predictions))
        }
        
        quality_stats[task_name] = stats
        
        # Log analysis
        logger.info(f"{task_name} predictions:")
        logger.info(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
        logger.info(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        logger.info(f"  Unique values: {stats['unique_values']}")
        
        # Quality warnings
        if stats['std'] < 1e-8:
            logger.warning(f"  ⚠️  CONSTANT PREDICTIONS for {task_name}!")
        elif stats['unique_values'] < 5:
            logger.warning(f"  ⚠️  LIMITED PREDICTION DIVERSITY for {task_name}")
    
    logger.info("="*60)
    
    return quality_stats


def convert_targets_dict_to_array(targets: Dict[str, np.ndarray], 
                                task_names: List[str]) -> np.ndarray:
    """
    将字典格式的目标变量转换为数组格式
    
    Args:
        targets: Dictionary of target arrays
        task_names: List of task names in desired order
        
    Returns:
        Array of shape [samples, num_tasks]
    """
    target_arrays = []
    
    for task_name in task_names:
        if task_name in targets:
            target_arrays.append(targets[task_name])
        else:
            # Create zeros if task not found
            sample_size = len(list(targets.values())[0]) if targets else 1000
            target_arrays.append(np.zeros(sample_size, dtype=np.float32))
            logger.warning(f"Task {task_name} not found, using zeros")
    
    return np.column_stack(target_arrays).astype(np.float32)


def convert_array_to_targets_dict(target_array: np.ndarray, 
                                task_names: List[str]) -> Dict[str, np.ndarray]:
    """
    将数组格式的目标变量转换为字典格式
    
    Args:
        target_array: Array of shape [samples, num_tasks]
        task_names: List of task names corresponding to columns
        
    Returns:
        Dictionary of target arrays
    """
    targets = {}
    
    for i, task_name in enumerate(task_names):
        targets[task_name] = target_array[:, i] if target_array.ndim > 1 else target_array
    
    return targets