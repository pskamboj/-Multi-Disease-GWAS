"""
Prediction Script
Load trained models and make predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from data_loader import DataLoader


class MotifPredictor:
    """Make predictions using trained hybrid motif prediction models."""
    
    def __init__(self, density_model_path='models/density_model.pkl',
                 motif_model_path='models/motif_classifier.pkl',
                 metadata_path='models/metadata.pkl'):
        """
        Initialize predictor with trained models.
        
        Args:
            density_model_path: Path to saved density prediction model
            motif_model_path: Path to saved motif classification model
            metadata_path: Path to saved metadata
        """
        print("Loading models...")
        self.density_model = joblib.load(density_model_path)
        self.motif_classifier = joblib.load(motif_model_path)
        
        metadata = joblib.load(metadata_path)
        self.density_threshold = metadata['density_threshold']
        self.feature_columns = metadata['feature_columns']
        self.motif_columns = metadata['motif_columns']
        self.best_model_name = metadata['best_model_name']
        
        print(f"Loaded {self.best_model_name} for density prediction")
        print(f"Density threshold: {self.density_threshold:.2f}")
        print(f"Number of features: {len(self.feature_columns)}")
        print(f"Number of motifs: {len(self.motif_columns)}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix (must have same features as training data)
            
        Returns:
            Dictionary with predictions
        """
        # Stage 1: Predict density
        density_pred = self.density_model.predict(X)
        
        # Stage 2: Predict individual motifs for high-density cases
        high_density_mask = density_pred >= self.density_threshold
        motif_predictions = np.zeros((len(X), len(self.motif_columns)))
        
        if np.any(high_density_mask):
            X_high = X[high_density_mask]
            motif_predictions[high_density_mask] = self.motif_classifier.predict(X_high)
        
        return {
            'density': density_pred,
            'motifs': motif_predictions,
            'high_density_mask': high_density_mask,
            'motif_columns': self.motif_columns
        }
    
    def predict_from_file(self, input_file, output_file=None):
        """
        Make predictions from a CSV file.
        
        Args:
            input_file: Path to input CSV file (same format as training data)
            output_file: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        print(f"\nLoading data from {input_file}...")
        
        # Load and preprocess data
        loader = DataLoader(input_file)
        loader.load_data()
        X = loader.preprocess_features()
        
        # Ensure features match training data
        if set(X.columns) != set(self.feature_columns):
            print("Warning: Feature mismatch. Aligning features...")
            # Add missing columns with zeros
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            # Remove extra columns
            X = X[self.feature_columns]
        
        print(f"Making predictions on {len(X)} samples...")
        
        # Make predictions
        predictions = self.predict(X)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'predicted_motif_density': predictions['density'],
            'is_high_density': predictions['high_density_mask']
        })
        
        # Add individual motif predictions
        motif_pred_df = pd.DataFrame(
            predictions['motifs'],
            columns=self.motif_columns
        )
        results = pd.concat([results, motif_pred_df], axis=1)
        
        # Add original data columns for reference
        original_cols = ['chrom', 'chromStart', 'chromEnd', 'genes', 'trait', 'final_disease']
        for col in original_cols:
            if col in loader.df.columns:
                results[col] = loader.df[col].values
        
        # Reorder columns
        info_cols = [col for col in results.columns if col in original_cols]
        pred_cols = ['predicted_motif_density', 'is_high_density']
        motif_cols = [col for col in results.columns if col.startswith('motif_')]
        results = results[info_cols + pred_cols + motif_cols]
        
        # Save to file if specified
        if output_file:
            results.to_csv(output_file, index=False)
            print(f"\nPredictions saved to {output_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("PREDICTION SUMMARY")
        print("="*70)
        print(f"Total samples: {len(results)}")
        print(f"High-density samples: {predictions['high_density_mask'].sum()} ({100*predictions['high_density_mask'].mean():.1f}%)")
        print(f"Average predicted density: {predictions['density'].mean():.2f}")
        print(f"Density range: [{predictions['density'].min():.2f}, {predictions['density'].max():.2f}]")
        
        if predictions['high_density_mask'].sum() > 0:
            avg_motifs_per_sample = predictions['motifs'][predictions['high_density_mask']].sum(axis=1).mean()
            print(f"Average motifs per high-density sample: {avg_motifs_per_sample:.1f}")
        
        print("="*70)
        
        return results
    
    def predict_single_mutation(self, mutation_data):
        """
        Make prediction for a single mutation.
        
        Args:
            mutation_data: Dictionary with mutation information
            
        Returns:
            Dictionary with predictions
        """
        # Convert to DataFrame
        df = pd.DataFrame([mutation_data])
        
        # Preprocess
        loader = DataLoader('')
        loader.df = df
        X = loader.preprocess_features()
        
        # Align features
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns]
        
        # Predict
        predictions = self.predict(X)
        
        result = {
            'predicted_density': float(predictions['density'][0]),
            'is_high_density': bool(predictions['high_density_mask'][0]),
            'predicted_motifs': {}
        }
        
        # Add motif predictions for high-density cases
        if predictions['high_density_mask'][0]:
            for i, motif in enumerate(self.motif_columns):
                if predictions['motifs'][0, i] == 1:
                    result['predicted_motifs'][motif] = 1
        
        return result


def main():
    """Command-line interface for making predictions."""
    parser = argparse.ArgumentParser(description='Make motif predictions using trained models')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file')
    parser.add_argument('--density-model', type=str, default='models/density_model.pkl', 
                       help='Path to density model')
    parser.add_argument('--motif-model', type=str, default='models/motif_classifier.pkl',
                       help='Path to motif classifier')
    parser.add_argument('--metadata', type=str, default='models/metadata.pkl',
                       help='Path to metadata')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MotifPredictor(
        density_model_path=args.density_model,
        motif_model_path=args.motif_model,
        metadata_path=args.metadata
    )
    
    # Make predictions
    results = predictor.predict_from_file(args.input, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
