"""
Model Explainability using SHAP (SHapley Additive exPlanations)
Provides interpretable explanations for individual predictions
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class ModelExplainer:
    """
    Production-grade explainability wrapper
    Generates SHAP values and visualizations for model interpretability
    """
    
    def __init__(self, model, feature_names: List[str], background_samples: int = 100):
        """
        Args:
            model: Trained Keras model
            feature_names: List of feature names
            background_samples: Number of samples for SHAP background dataset
        """
        self.model = model
        self.feature_names = feature_names
        self.background_samples = background_samples
        self.explainer = None
        self.shap_values = None
        
    def initialize_explainer(self, X_background: np.ndarray):
        """
        Initialize SHAP Deep Explainer with background dataset
        Background data represents "typical" inputs for baseline comparison
        """
        # Use a subset for computational efficiency
        if len(X_background) > self.background_samples:
            indices = np.random.choice(
                len(X_background), 
                self.background_samples, 
                replace=False
            )
            background = X_background[indices]
        else:
            background = X_background
        
        print(f"Initializing SHAP explainer with {len(background)} background samples...")
        
        # DeepExplainer for neural networks
        self.explainer = shap.DeepExplainer(self.model, background)
        
        print("✓ SHAP explainer initialized")
    
    def explain_predictions(self, X: np.ndarray, max_samples: int = 100) -> np.ndarray:
        """
        Calculate SHAP values for given samples
        SHAP values represent feature importance for each prediction
        
        Returns:
            SHAP values array of shape [samples, sequence_length, features, classes]
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        # Limit samples for computational efficiency
        X_explain = X[:max_samples] if len(X) > max_samples else X
        
        print(f"Computing SHAP values for {len(X_explain)} samples...")
        self.shap_values = self.explainer.shap_values(X_explain)
        
        print("✓ SHAP values computed")
        return self.shap_values
    
    def get_feature_importance(self, class_idx: int = 2) -> pd.DataFrame:
        """
        Aggregate SHAP values to get global feature importance
        
        Args:
            class_idx: Class to explain (0=Down, 1=Neutral, 2=Up)
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_predictions() first.")
        
        # SHAP values shape: [samples, sequence_length, features, classes]
        # Average over samples and time steps
        shap_class = self.shap_values[class_idx]  # Get values for specific class
        
        # Calculate mean absolute SHAP value for each feature
        importance = np.mean(np.abs(shap_class), axis=(0, 1))
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, 
                               class_idx: int = 2, 
                               top_n: int = 20,
                               save_path: str = None) -> plt.Figure:
        """
        Visualize top N most important features
        
        Args:
            class_idx: Class to explain (0=Down, 1=Neutral, 2=Up)
            top_n: Number of top features to display
            save_path: Optional path to save figure
        """
        importance_df = self.get_feature_importance(class_idx)
        top_features = importance_df.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#e74c3c' if imp < 0 else '#3498db' for imp in top_features['importance']]
        
        ax.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features\nClass: {"Down" if class_idx==0 else "Neutral" if class_idx==1 else "Up"}', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to {save_path}")
        
        return fig
    
    def explain_single_prediction(self, 
                                 X_sample: np.ndarray,
                                 sample_idx: int = 0,
                                 class_idx: int = 2) -> Dict:
        """
        Generate detailed explanation for a single prediction
        
        Returns:
            Dictionary with prediction details and SHAP values
        """
        # Get prediction
        prediction = self.model.predict(X_sample[sample_idx:sample_idx+1], verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Get SHAP values for this sample
        if self.shap_values is None or len(self.shap_values[0]) <= sample_idx:
            # Compute if not already done
            self.explain_predictions(X_sample[sample_idx:sample_idx+1])
        
        shap_sample = self.shap_values[class_idx][sample_idx]  # [sequence_length, features]
        
        # Get most influential features (averaged over time)
        feature_shap_mean = np.mean(np.abs(shap_sample), axis=0)
        top_feature_indices = np.argsort(feature_shap_mean)[-5:][::-1]
        
        explanation = {
            'predicted_class': int(predicted_class),
            'predicted_class_name': ['Down', 'Neutral', 'Up'][predicted_class],
            'confidence': float(confidence),
            'class_probabilities': {
                'Down': float(prediction[0][0]),
                'Neutral': float(prediction[0][1]),
                'Up': float(prediction[0][2])
            },
            'top_5_features': [
                {
                    'feature': self.feature_names[idx],
                    'shap_value': float(feature_shap_mean[idx])
                }
                for idx in top_feature_indices
            ]
        }
        
        return explanation
    
    def plot_attention_heatmap(self, 
                              attention_weights: np.ndarray,
                              sample_idx: int = 0,
                              save_path: str = None) -> plt.Figure:
        """
        Visualize attention weights to show which time steps model focuses on
        
        Args:
            attention_weights: Attention weights from model [samples, time_steps, 1]
            sample_idx: Index of sample to visualize
            save_path: Optional path to save figure
        """
        weights = attention_weights[sample_idx].squeeze()
        
        fig, ax = plt.subplots(figsize=(12, 3))
        
        # Create heatmap
        im = ax.imshow(weights.reshape(1, -1), 
                      cmap='YlOrRd', 
                      aspect='auto',
                      interpolation='nearest')
        
        ax.set_yticks([])
        ax.set_xlabel('Time Step (Most Recent →)', fontsize=12)
        ax.set_title('Attention Weights: Which Time Steps Does the Model Focus On?', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
        cbar.set_label('Attention Weight', fontsize=10)
        
        # Highlight top 5 attention weights
        top_indices = np.argsort(weights)[-5:]
        for idx in top_indices:
            ax.axvline(x=idx, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Attention heatmap saved to {save_path}")
        
        return fig
    
    def generate_explanation_report(self, 
                                   X_sample: np.ndarray,
                                   attention_weights: np.ndarray,
                                   sample_idx: int = 0,
                                   output_dir: str = "explanations") -> Dict:
        """
        Generate comprehensive explanation report with visualizations
        
        Returns:
            Dictionary with all explanation components
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating explanation report for sample {sample_idx}...")
        
        # Get prediction explanation
        explanation = self.explain_single_prediction(X_sample, sample_idx)
        
        # Generate visualizations
        # 1. Feature importance
        self.plot_feature_importance(
            class_idx=explanation['predicted_class'],
            save_path=f"{output_dir}/feature_importance_sample_{sample_idx}.png"
        )
        plt.close()
        
        # 2. Attention heatmap
        self.plot_attention_heatmap(
            attention_weights,
            sample_idx=sample_idx,
            save_path=f"{output_dir}/attention_weights_sample_{sample_idx}.png"
        )
        plt.close()
        
        # Save explanation as JSON
        import json
        with open(f"{output_dir}/explanation_sample_{sample_idx}.json", 'w') as f:
            json.dump(explanation, f, indent=2)
        
        print(f"✓ Explanation report saved to {output_dir}/")
        print(f"  - Prediction: {explanation['predicted_class_name']} ({explanation['confidence']:.2%} confidence)")
        print(f"  - Top feature: {explanation['top_5_features'][0]['feature']}")
        
        return explanation


# Example usage
if __name__ == "__main__":
    from attention_lstm import StockPredictorModel
    
    # Load trained model (example)
    model = StockPredictorModel(sequence_length=60, n_features=40)
    model.build_model()
    
    # Dummy data for demonstration
    X_background = np.random.randn(200, 60, 40)
    X_test = np.random.randn(50, 60, 40)
    
    feature_names = [f"feature_{i}" for i in range(40)]
    
    # Initialize explainer
    explainer = ModelExplainer(model.model, feature_names)
    explainer.initialize_explainer(X_background)
    
    # Compute SHAP values
    shap_values = explainer.explain_predictions(X_test, max_samples=20)
    
    # Get feature importance
    importance = explainer.get_feature_importance(class_idx=2)
    print("\nTop 10 Features:")
    print(importance.head(10))
    
    # Generate explanation report
    attention_weights = model.get_attention_weights(X_test)
    report = explainer.generate_explanation_report(
        X_test, 
        attention_weights,
        sample_idx=0
    )
    
    print("\n✓ Explainability pipeline completed!")