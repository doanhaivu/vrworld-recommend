# Require charts

user concentraion, world concentration chart, action type 
distribution, session duration distribution, engagement pattern, world
categories performance chart, interaction matrix sample, user exploration
pattern, world reach distribution, user embeddings (ALS, t-SNE projection), top
feature correlations, feature correlation matrix chart, also comprehensive model
	comparision with ROC AUC score, average precision, ranking metrics @10, model
performance radar having precision@10, avg precision, ROC AUC, recall@10,
NDCG@10 charts...

# Running the notebook

/opt/anaconda3/bin/jupyter nbconvert --to notebook --execute recommendation_combined.ipynb --output recommendation_combined_executed.ipynb


# Installing
/opt/anaconda3/bin/pip install seaborn plotly implicit


#


✅ Fixed Issues:

1. Data Leakage Prevention

- Properly excluded problematic columns (action_id,
action_timestamp)
- Added explicit validation to ensure excluded columns are
removed after merging
- Created clean feature lists with only safe features

2. Negative Sampling Fix

- Implemented efficient dictionary-based lookup for
existing interactions
- Guaranteed non-overlap with existing user-world pairs
- Added maximum attempts safety limit

3. Missing Data Handling

- Separated categorical and numerical features
- Used appropriate imputation strategies (mode for
categorical, median for numerical)
- Added missing value handling before encoding

4. Standardized Evaluation

- Both models now use 80/20 train-test split
- ALS evaluation matches Random Forest methodology
- Added average precision metric for ALS
- Fair comparison with same evaluation approach

5. Simulated Data Labels

- Added clear "⚠️ SIMULATED DATA" warnings to all simulated
	visualizations
- Added explanatory notes about which data is real vs
simulated

6. Error Handling

- Added comprehensive file validation
- Added column existence checks
- Added empty dataframe handling
- Added try-catch blocks for critical operations

7. Performance Optimizations

- Replaced inefficient set operations with dictionary
lookups
- Optimized negative sampling algorithm
- Added early termination conditions

8. Statistical Calculations

- Fixed Gini coefficient calculation with proper formula
- Added interpretation of Gini values
- Fixed correlation analysis to account for class imbalance

The notebook is now production-ready with robust error
handling, no data leakage, fair model comparison, and clear
	documentation of any simulated data used for demonstration
	purposes.