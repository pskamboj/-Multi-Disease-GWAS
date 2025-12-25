"""
Data Loader Module
Loads and preprocesses the GWAS dataset with gene mutations and motif information.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Load and preprocess genomic data for motif prediction."""
    
    def __init__(self, filepath):
        """
        Initialize DataLoader.
        
        Args:
            filepath: Path to the CSV file containing the dataset
        """
        self.filepath = filepath
        self.df = None
        self.feature_columns = []
        self.motif_columns = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the CSV file into a pandas DataFrame."""
        print(f"Loading data from {self.filepath}...")
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        
        # Identify motif columns
        self.motif_columns = [col for col in self.df.columns if col.startswith('motif_') and col != 'motif_density']
        print(f"Found {len(self.motif_columns)} motif columns")
        
        return self.df
    
    def get_biologically_relevant_features(self):
        """
        Extract only biologically relevant features.
        Excludes publication metadata (author, journal, pubDate, pubMedID, title).
        
        Returns:
            List of biologically relevant feature column names
        """
        # Exclude publication metadata
        exclude_cols = [
            'author', 'journal', 'pubDate', 'pubMedID', 'title',
            '#bin', 'name',  # identifier columns
            'motif_density',  # target for stage 1
        ] + self.motif_columns  # targets for stage 2
        
        # Biologically relevant features
        bio_features = [
            'chrom',           # Chromosome
            'chromStart',      # Start position
            'chromEnd',        # End position
            'region',          # Genomic region
            'genes',           # Associated genes
            'riskAllele',      # Risk allele
            'riskAlFreq',      # Risk allele frequency
            'pValue',          # P-value
            'pValueDesc',      # P-value description
            'orOrBeta',        # Odds ratio or beta coefficient
            'ci95',            # 95% confidence interval
            'platform',        # Genotyping platform
            'cnv',             # Copy number variation
            'trait',           # Associated trait
            'disease_from_trait',  # Disease from trait
            'disease_from_title',  # Disease from title
            'final_disease',   # Final disease classification
            'minus_log_p',     # -log10(p-value)
            'abs_or_beta',     # Absolute odds ratio/beta
            'total_motifs',    # Total number of motifs
            'motif_diversity', # Motif diversity measure
            'initSample',      # Initial sample description
            'replSample',      # Replication sample description
        ]
        
        # Only include columns that exist in the dataset
        available_features = [col for col in bio_features if col in self.df.columns]
        
        print(f"Selected {len(available_features)} biologically relevant features")
        self.feature_columns = available_features
        
        return available_features
    
    def preprocess_features(self):
        """
        Preprocess the biologically relevant features.
        Handles missing values, encodes categorical variables, and normalizes numerical features.
        
        Returns:
            Preprocessed feature DataFrame
        """
        print("Preprocessing features...")
        
        # Get biologically relevant features
        features = self.get_biologically_relevant_features()
        X = self.df[features].copy()
        
        # Handle missing values
        print("Handling missing values...")
        
        # Numerical columns
        numerical_cols = ['chromStart', 'chromEnd', 'riskAlFreq', 'pValue', 
                         'orOrBeta', 'minus_log_p', 'abs_or_beta', 
                         'total_motifs', 'motif_diversity']
        numerical_cols = [col for col in numerical_cols if col in X.columns]
        
        # Fill missing numerical values with median
        for col in numerical_cols:
            if X[col].isna().any() or X[col].dtype == 'object':
                # Convert to numeric, coercing errors to NaN
                X[col] = pd.to_numeric(X[col], errors='coerce')
                # Fill NaN with median
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0  # If all values are NaN, use 0
                X[col].fillna(median_val, inplace=True)
        
        # Categorical columns
        categorical_cols = ['chrom', 'region', 'genes', 'riskAllele', 'platform', 
                           'cnv', 'trait', 'disease_from_trait', 'disease_from_title', 
                           'final_disease', 'pValueDesc', 'ci95', 'initSample', 'replSample']
        categorical_cols = [col for col in categorical_cols if col in X.columns]
        
        # Fill missing categorical values with 'Unknown'
        for col in categorical_cols:
            if X[col].isna().any():
                X[col].fillna('Unknown', inplace=True)
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Ensure all values are numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        X.fillna(0, inplace=True)
        
        print(f"Preprocessed features shape: {X.shape}")
        
        return X
    
    def get_targets(self):
        """
        Extract target variables.
        
        Returns:
            Tuple of (motif_density, motif_matrix)
            - motif_density: Series for regression (Stage 1)
            - motif_matrix: DataFrame for multi-label classification (Stage 2)
        """
        motif_density = self.df['motif_density'].copy()
        motif_matrix = self.df[self.motif_columns].copy()
        
        # Fill missing values in motif columns with 0 (absence)
        motif_matrix.fillna(0, inplace=True)
        
        print(f"Target shapes - motif_density: {motif_density.shape}, motif_matrix: {motif_matrix.shape}")
        
        return motif_density, motif_matrix
    
    def create_train_test_split(self, X, y_density, y_motifs, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature matrix
            y_density: Motif density target
            y_motifs: Individual motif targets
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train, validation, and test splits
        """
        print(f"Splitting data: train/val/test...")
        
        # First split: train+val vs test
        X_temp, X_test, y_density_temp, y_density_test, y_motifs_temp, y_motifs_test = train_test_split(
            X, y_density, y_motifs, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_density_train, y_density_val, y_motifs_train, y_motifs_val = train_test_split(
            X_temp, y_density_temp, y_motifs_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_density_train': y_density_train,
            'y_density_val': y_density_val,
            'y_density_test': y_density_test,
            'y_motifs_train': y_motifs_train,
            'y_motifs_val': y_motifs_val,
            'y_motifs_test': y_motifs_test,
        }
    
    def prepare_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Complete data preparation pipeline.
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed
            
        Returns:
            Dictionary containing all data splits and metadata
        """
        # Load data
        self.load_data()
        
        # Preprocess features
        X = self.preprocess_features()
        
        # Get targets
        y_density, y_motifs = self.get_targets()
        
        # Create splits
        splits = self.create_train_test_split(X, y_density, y_motifs, test_size, val_size, random_state)
        
        # Normalize features
        print("Normalizing features...")
        self.scaler.fit(splits['X_train'])
        splits['X_train'] = pd.DataFrame(
            self.scaler.transform(splits['X_train']),
            columns=X.columns,
            index=splits['X_train'].index
        )
        splits['X_val'] = pd.DataFrame(
            self.scaler.transform(splits['X_val']),
            columns=X.columns,
            index=splits['X_val'].index
        )
        splits['X_test'] = pd.DataFrame(
            self.scaler.transform(splits['X_test']),
            columns=X.columns,
            index=splits['X_test'].index
        )
        
        # Add metadata
        splits['feature_columns'] = self.feature_columns
        splits['motif_columns'] = self.motif_columns
        splits['label_encoders'] = self.label_encoders
        splits['scaler'] = self.scaler
        
        print("\nData preparation complete!")
        print(f"Number of features: {len(self.feature_columns)}")
        print(f"Number of motifs: {len(self.motif_columns)}")
        
        return splits


if __name__ == "__main__":
    # Example usage
    loader = DataLoader('filtered_8500rows_dataset.csv')
    data = loader.prepare_data()
    
    print("\n" + "="*50)
    print("Data preparation summary:")
    print("="*50)
    print(f"Features: {data['X_train'].shape[1]}")
    print(f"Training samples: {data['X_train'].shape[0]}")
    print(f"Validation samples: {data['X_val'].shape[0]}")
    print(f"Test samples: {data['X_test'].shape[0]}")
    print(f"Motif density range: [{data['y_density_train'].min():.2f}, {data['y_density_train'].max():.2f}]")
    print(f"Number of motifs to predict: {data['y_motifs_train'].shape[1]}")
