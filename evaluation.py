"""
Evaluation Module
Provides comprehensive evaluation metrics and visualizations for the hybrid motif prediction model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report,
    hamming_loss, f1_score, jaccard_score
)
import os


class ModelEvaluator:
    """Evaluate and visualize model performance."""
    
    def __init__(self, output_dir='results'):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_regression(self, y_true, y_pred, model_name='Model', save_plots=True):
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_plots: Whether to save visualization plots
            
        Returns:
            Dictionary of metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"\n{model_name} - Regression Metrics:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        if save_plots:
            self._plot_regression_results(y_true, y_pred, model_name)
        
        return metrics
    
    def _plot_regression_results(self, y_true, y_pred, model_name):
        """Create regression visualization plots."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot: Actual vs Predicted
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Motif Density', fontsize=12)
        axes[0].set_ylabel('Predicted Motif Density', fontsize=12)
        axes[0].set_title(f'{model_name} - Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Motif Density', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title(f'{model_name} - Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{model_name.lower().replace(" ", "_")}_regression.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot: {self.output_dir}/{model_name.lower().replace(' ', '_')}_regression.png")
    
    def evaluate_multilabel(self, y_true, y_pred, motif_columns, model_name='Model', save_plots=True):
        """
        Evaluate multi-label classification performance.
        
        Args:
            y_true: True labels (n_samples, n_motifs)
            y_pred: Predicted labels (n_samples, n_motifs)
            motif_columns: List of motif column names
            model_name: Name of the model
            save_plots: Whether to save visualization plots
            
        Returns:
            Dictionary of metrics
        """
        hamming = hamming_loss(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        jaccard = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
        
        metrics = {
            'Hamming_Loss': hamming,
            'F1_Micro': f1_micro,
            'F1_Macro': f1_macro,
            'Jaccard_Score': jaccard
        }
        
        print(f"\n{model_name} - Multi-Label Classification Metrics:")
        print(f"  Hamming Loss:     {hamming:.4f}")
        print(f"  F1-Score (Micro): {f1_micro:.4f}")
        print(f"  F1-Score (Macro): {f1_macro:.4f}")
        print(f"  Jaccard Score:    {jaccard:.4f}")
        
        if save_plots:
            self._plot_multilabel_results(y_true, y_pred, motif_columns, model_name)
        
        return metrics
    
    def _plot_multilabel_results(self, y_true, y_pred, motif_columns, model_name):
        """Create multi-label classification visualization plots."""
        # Calculate per-motif F1 scores
        f1_scores = []
        for i in range(y_true.shape[1]):
            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            f1_scores.append(f1)
        
        # Sort by F1 score
        sorted_indices = np.argsort(f1_scores)[::-1]
        
        # Plot top 20 and bottom 20 motifs
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top 20
        top_20_indices = sorted_indices[:20]
        top_20_motifs = [motif_columns[i] for i in top_20_indices]
        top_20_scores = [f1_scores[i] for i in top_20_indices]
        
        axes[0].barh(range(20), top_20_scores, color='green', alpha=0.7)
        axes[0].set_yticks(range(20))
        axes[0].set_yticklabels(top_20_motifs, fontsize=9)
        axes[0].set_xlabel('F1-Score', fontsize=12)
        axes[0].set_title(f'{model_name} - Top 20 Motifs by F1-Score', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].invert_yaxis()
        
        # Bottom 20
        bottom_20_indices = sorted_indices[-20:]
        bottom_20_motifs = [motif_columns[i] for i in bottom_20_indices]
        bottom_20_scores = [f1_scores[i] for i in bottom_20_indices]
        
        axes[1].barh(range(20), bottom_20_scores, color='red', alpha=0.7)
        axes[1].set_yticks(range(20))
        axes[1].set_yticklabels(bottom_20_motifs, fontsize=9)
        axes[1].set_xlabel('F1-Score', fontsize=12)
        axes[1].set_title(f'{model_name} - Bottom 20 Motifs by F1-Score', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{model_name.lower().replace(" ", "_")}_motif_f1.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot: {self.output_dir}/{model_name.lower().replace(' ', '_')}_motif_f1.png")
        
        # Save detailed per-motif metrics to CSV
        motif_metrics = pd.DataFrame({
            'Motif': motif_columns,
            'F1_Score': f1_scores,
            'True_Positives': [np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1)) for i in range(y_true.shape[1])],
            'False_Positives': [np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1)) for i in range(y_true.shape[1])],
            'False_Negatives': [np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0)) for i in range(y_true.shape[1])],
            'True_Negatives': [np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0)) for i in range(y_true.shape[1])]
        })
        motif_metrics = motif_metrics.sort_values('F1_Score', ascending=False)
        motif_metrics.to_csv(f'{self.output_dir}/motif_performance_details.csv', index=False)
        
        print(f"  Saved detailed metrics: {self.output_dir}/motif_performance_details.csv")
    
    def plot_feature_importance(self, model, feature_names, model_name='Model', top_n=20):
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to display
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices], color='steelblue', alpha=0.7)
        plt.yticks(range(top_n), [feature_names[i] for i in indices], fontsize=10)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'{model_name} - Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{model_name.lower().replace(" ", "_")}_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved feature importance plot: {self.output_dir}/{model_name.lower().replace(' ', '_')}_feature_importance.png")
        
        # Save to CSV
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        importance_df.to_csv(f'{self.output_dir}/feature_importance.csv', index=False)
        print(f"  Saved feature importance data: {self.output_dir}/feature_importance.csv")
    
    def create_summary_report(self, stage1_metrics, stage2_metrics, model_info):
        """
        Create a comprehensive summary report.
        
        Args:
            stage1_metrics: Dictionary of Stage 1 (regression) metrics
            stage2_metrics: Dictionary of Stage 2 (classification) metrics
            model_info: Dictionary with model metadata
        """
        report_path = f'{self.output_dir}/evaluation_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("HYBRID MOTIF PREDICTION - EVALUATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("MODEL INFORMATION\n")
            f.write("-"*70 + "\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("STAGE 1: MOTIF DENSITY PREDICTION (REGRESSION)\n")
            f.write("-"*70 + "\n")
            for metric, value in stage1_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            f.write("STAGE 2: INDIVIDUAL MOTIF PREDICTION (MULTI-LABEL CLASSIFICATION)\n")
            f.write("-"*70 + "\n")
            for metric, value in stage2_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
        
        print(f"\nSummary report saved: {report_path}")


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator(output_dir='results')
    
    # Simulate some predictions for demonstration
    np.random.seed(42)
    y_true_reg = np.random.rand(100) * 100
    y_pred_reg = y_true_reg + np.random.randn(100) * 10
    
    y_true_multi = np.random.randint(0, 2, (100, 50))
    y_pred_multi = y_true_multi.copy()
    # Create a single mask to avoid size mismatch
    mask = np.random.rand(100, 50) > 0.8
    y_pred_multi[mask] = 1 - y_pred_multi[mask]
    
    motif_cols = [f'motif_{i}' for i in range(50)]
    
    evaluator.evaluate_regression(y_true_reg, y_pred_reg, 'Test Model')
    evaluator.evaluate_multilabel(y_true_multi, y_pred_multi, motif_cols, 'Test Model')
