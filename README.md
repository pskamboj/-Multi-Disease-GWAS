# Regulatory Variant Disease Association Prediction

A machine learning and Bayesian framework for predicting disease associations from regulatory GWAS variants using transcription factor motif disruption, effect sizes, and statistical significance.

## Overview

This project analyzes a large-scale GWAS dataset containing regulatory variants annotated with transcription factor (TF) motif matches. The goal is to predict the specific disease or trait associated with each variant by combining:

- GWAS effect size (`orOrBeta`)
- Statistical significance (`pValue`)
- Transcription factor motif disruption patterns

The pipeline includes data cleaning, multi-class disease labeling, biologically informed feature engineering, and multiple importance scoring techniques for motif-disease associations.

Key components:
- Multi-class classification across dozens of diseases and traits
- Composite motif importance scoring using multiple statistical measures
- Handling of severe class imbalance
- Comprehensive evaluation and visualization of motif-disease relationships

## Dataset

The input consists of GWAS variants with the following key columns:
- `trait` / `title`: Disease or trait name
- `orOrBeta`: Odds ratio or regression coefficient
- `pValue`: Association p-value
- `motif_*`: Continuous or binary scores indicating TF binding site disruption
- Additional metadata (chromosome, position, genes, etc.)

After processing, unknown or low-frequency traits are filtered, resulting in a clean multi-class dataset with known disease labels.

## Methods

### Disease Labeling
Diseases and traits are extracted and standardized from the `trait` and `title` columns using a comprehensive mapping dictionary covering major cancers, autoimmune, cardiovascular, metabolic, neurological, and anthropometric traits.

### Feature Engineering
- `-log10(pValue)` transformation
- Absolute effect size (`|orOrBeta|`)
- Motif count and diversity summaries

### Motif Importance Scoring
Six complementary techniques are computed for each motif-disease pair:
- TF-IDF style scoring
- Normalized lift
- Balanced mutual information
- Odds ratio with confidence intervals
- Normalized chi-square statistic
- Predictive F1-score contribution

A weighted composite score combines all methods for robust ranking.

### Evaluation
Results include per-class performance metrics and top motif associations for biological interpretation.

## Output Files
- `motif_importance_all_techniques.csv`: Full table with all scoring methods
- `top_motifs_per_class_*.csv`: Top motifs ranked by different scores
- Visualization plots (heatmaps, bar plots, network graphs)

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn
- networkx

Install dependencies with:
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn networkx
