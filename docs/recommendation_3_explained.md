# Understanding Recommendation Systems: ALS vs Random Forest
## A Technical Deep-Dive for CS Graduates

## 1. Overview: The Recommendation Challenge

This notebook tackles a fundamental problem in recommendation systems: **How do we predict which items users will like?** It compares two radically different approaches:

1. **ALS (Alternating Least Squares)** - Collaborative Filtering
2. **Random Forest** - Content-Based Classification

Think of it like Netflix trying to recommend movies: Should they look at what similar users watched (collaborative) or analyze movie features like genre and actors (content-based)?

## 2. Data Preprocessing: Building the Foundation

### 2.1 Data Loading and Normalization
```python
users = pd.read_csv('users.csv', dtype=str)
worlds = pd.read_csv('vrworld.csv', dtype=str)
actions = pd.read_csv('user_actions.csv', dtype=str)
time_spent = pd.read_csv('time_spent.csv', dtype=str)
```

The data includes:
- **5,000 users** with demographics
- **400 virtual worlds** with metadata
- **40,000 actions** (user interactions)
- **20,000 time records** (engagement metrics)

### 2.2 Label Encoding: Converting IDs to Integers
```python
user_encoder = LabelEncoder().fit(users['user_id'])
world_encoder = LabelEncoder().fit(worlds['world_id'])
actions['user_idx'] = user_encoder.transform(actions['user_id'])
```

**Why Label Encoding?**
- String IDs like "U123456" are inefficient for matrix operations
- Integer indices [0, 1, 2, ...] enable array-based operations
- Maintains bidirectional mapping for decoding results

### 2.3 Creating the Interaction Matrix
```python
merged['weight'] = merged['actions'] + merged['minutes'].fillna(0)/merged['minutes'].fillna(0).max()
interaction_matrix = sparse.csr_matrix((merged['weight'], 
                                       (merged['user_idx'], merged['world_idx'])), 
                                       shape=(n_users, n_worlds))
```

**The Weight Formula:**
```
weight = action_count + (time_spent / max_time)
```

This creates an **implicit feedback signal**:
- More actions → Higher weight
- More time spent → Higher weight
- Combines frequency and engagement

**Sparse Matrix Format (CSR):**
- Full matrix: 5,000 × 400 = 2,000,000 cells
- Non-zero entries: 39,592 (only 2% density!)
- CSR format stores only non-zero values, saving ~98% memory

## 3. Train-Test Split: Leave-One-Out Strategy

### 3.1 The Algorithm
```python
def train_test_split_interactions(interactions_df, seed=RANDOM_STATE):
    for uid, g in interactions_df.groupby('user_idx'):
        if len(idxs) == 1:
            train_rows.append(idxs[0])  # Keep in training
        else:
            test_idx = rng.choice(idxs)  # Random one for testing
            # Rest go to training
```

**Leave-One-Out Logic:**
- For each user with multiple interactions, hold out ONE random interaction for testing
- Keep remaining interactions for training
- Users with only one interaction stay in training (no test data)

**Why This Strategy?**
- Ensures every user (with 2+ interactions) has test data
- Maintains temporal consistency
- Realistic: Predicting the "next" item a user might like

## 4. Evaluation Metrics: Beyond Accuracy

### 4.1 Classification Metrics

**ROC-AUC (Area Under ROC Curve):**
```python
roc_auc_score(labels, scores)
```

Measures how well the model ranks positive items above negative items:
- AUC = 1.0: Perfect ranking
- AUC = 0.5: Random guessing
- AUC < 0.5: Worse than random

**Average Precision (PR-AUC):**
```python
average_precision_score(labels, scores)
```

Similar to ROC-AUC but focuses on positive class precision - important when data is imbalanced.

### 4.2 Ranking Metrics

**Precision@k:**
```python
precision = y_topk.sum() / k
```

Mathematics:
```
Precision@k = |relevant items in top-k| / k
```

Example: If 3 out of top-10 recommendations are relevant:
```
Precision@10 = 3/10 = 0.3
```

**Recall@k:**
```python
recall = y_topk.sum() / max(1, y_true.sum())
```

Mathematics:
```
Recall@k = |relevant items in top-k| / |all relevant items|
```

Example: If user has 20 relevant items total and we find 3 in top-10:
```
Recall@10 = 3/20 = 0.15
```

**NDCG@k (Normalized Discounted Cumulative Gain):**
```python
gains = (2**y_topk - 1) / np.log2(np.arange(2, len(y_topk) + 2))
dcg = gains.sum()
```

The formula:
```
DCG@k = Σᵢ₌₁ᵏ (2^relᵢ - 1) / log₂(i + 1)
NDCG@k = DCG@k / IDCG@k
```

Where:
- relᵢ = relevance of item at position i (0 or 1 in our case)
- IDCG = DCG of ideal ranking (all relevant items first)
- Higher positions get more weight (1/log discount factor)

## 5. ALS (Alternating Least Squares): Collaborative Filtering

### 5.1 The Mathematics Behind ALS

ALS performs **matrix factorization**:
```
R ≈ U × Vᵀ
```

Where:
- R = User-Item interaction matrix (5000 × 400)
- U = User factors matrix (5000 × 64)
- V = Item factors matrix (400 × 64)
- 64 = number of latent factors

### 5.2 The Optimization Problem

ALS minimizes:
```
L = Σᵤ,ᵢ cᵤᵢ(rᵤᵢ - uᵤᵀvᵢ)² + λ(||uᵤ||² + ||vᵢ||²)
```

Where:
- cᵤᵢ = confidence weight (higher for observed interactions)
- rᵤᵢ = interaction value
- uᵤ = user u's factor vector
- vᵢ = item i's factor vector
- λ = regularization parameter (prevents overfitting)

### 5.3 The "Alternating" Part

The algorithm alternates between:
1. **Fix V, solve for U:** Each user's factors independently
2. **Fix U, solve for V:** Each item's factors independently

This converts a non-convex problem into alternating convex problems!

### 5.4 Making Predictions
```python
scores = model.user_factors[u] @ model.item_factors.T
```

This is a dot product:
```
score(user u, item i) = Σₖ (user_factor_uk × item_factor_ik)
```

Higher score = stronger predicted preference.

### 5.5 Why ALS Performs Poorly Here

**Results:** AUC = 0.46, Precision@10 = 0.0005

**Reasons:**
1. **Extreme sparsity:** Only 2% of matrix is filled
2. **Implicit feedback:** No explicit ratings, just interactions
3. **Cold start:** New items/users have no history
4. **Random data patterns:** Synthetic data lacks real collaborative signals

## 6. Random Forest: Content-Based Approach

### 6.1 Feature Engineering

The RF uses 26 features per user-item pair:
```python
X_enc.head()
# Features include: action_id, action_type, timestamp, user demographics,
# world properties (rating, likes, players), etc.
```

### 6.2 Creating Training Data

**Positive samples:** Actual interactions
```python
actions_enc['label'] = 1  # All real interactions are positive
```

**Negative sampling:**
```python
all_pairs = pd.MultiIndex.from_product([users['user_id'], worlds['world_id']])
neg = all_pairs.merge(actions_enc[['user_id','world_id']], 
                     how='left', indicator=True)
neg = neg[neg['_merge']=='left_only']  # Non-interactions
neg = neg.sample(len(actions_enc))      # Sample equal number
neg['label'] = 0
```

This creates a **balanced dataset**:
- 40,000 positive examples (real interactions)
- 40,000 negative examples (sampled non-interactions)

### 6.3 How Random Forest Works

**Decision Trees:**
Each tree makes splits like:
```
if user_age_group == "18-24":
    if world_category == "Gaming":
        prediction = 0.8  # High probability
    else:
        prediction = 0.3  # Lower probability
```

**Ensemble Learning:**
- 100 trees vote on final prediction
- Each tree sees random subset of features and samples
- Final score = average of all tree predictions

### 6.4 The Overfitting Problem

**Results:** AUC = 1.0, but Recall@10 = 0.0007

**Why perfect classification but terrible ranking?**

The model learns a trivial pattern:
```
if (user, item) in training_data:
    predict 1.0
else:
    predict based on features (poorly)
```

This works for classification because:
- Positive samples = actual interactions (have history)
- Negative samples = random non-interactions (no history)

But fails for recommendation because:
- All test items are "unseen" (not in training)
- Model can't distinguish between good and bad unseen items

## 7. The Comparison: Why Different Metrics Tell Different Stories

### 7.1 Results Summary
```
Model | AUC   | AP    | Prec@10 | Recall@10 | NDCG@10
------|-------|-------|---------|-----------|--------
ALS   | 0.464 | 0.009 | 0.0005  | 0.005     | 0.002
RF    | 1.000 | 1.000 | 1.000   | 0.0007    | 1.000
```

### 7.2 Interpretation

**ALS:**
- Poor classification (AUC < 0.5) - worse than random
- Poor ranking metrics - can't identify relevant items
- Struggling with sparse, implicit feedback

**Random Forest:**
- Perfect classification - but on wrong task!
- Terrible recall despite perfect precision
- The precision@10 = 1.0 is misleading (artifact of evaluation)

### 7.3 The Key Insight

**Classification ≠ Ranking**

Good recommendation systems need to:
1. Rank all items for a user
2. Put relevant items at the top
3. Work on unseen items

The RF excels at identifying training patterns but fails at generalizing to new recommendations.

## 8. Visualizations and Interpretability

### 8.1 t-SNE Visualization (ALS)
```python
user_factors_2d = TSNE(n_components=2).fit_transform(user_factors)
```

t-SNE reduces 64-dimensional user embeddings to 2D:
- Similar users cluster together
- Reveals user segments and preferences
- Can identify outliers and groups

### 8.2 Feature Importance (RF)
```python
importances = rf.feature_importances_
```

Shows which features drive predictions:
- Likely dominated by action-related features
- User/item IDs might have high importance (overfitting signal)
- Content features (genre, ratings) should matter but might not

## 9. Practical Takeaways

### 9.1 When to Use Each Approach

**Use ALS/Collaborative Filtering when:**
- You have dense interaction data
- User preferences are influenced by similar users
- You can handle the cold start problem

**Use Random Forest/Content-Based when:**
- You have rich item/user features
- Need to handle new items immediately
- Want interpretable recommendations

### 9.2 Real-World Improvements

1. **Hybrid approaches:** Combine collaborative and content signals
2. **Better negative sampling:** Use harder negatives (popular items user didn't interact with)
3. **Temporal splits:** Use time-based train/test splits
4. **Deep learning:** Neural collaborative filtering, transformers

### 9.3 The Data Quality Issue

Both models struggle because the synthetic data lacks:
- Real preference patterns
- Temporal dynamics
- Social influence signals
- Content-preference correlations

## 10. Code Optimization Notes

### Sparse Matrix Usage
```python
sparse.csr_matrix()  # Compressed Sparse Row format
```
- Memory efficient for 2% density
- Fast row operations (user lookups)
- Built-in support in scikit-learn and implicit

### Vectorized Operations
```python
scores = model.user_factors[u] @ model.item_factors.T
```
Uses BLAS for fast matrix multiplication instead of loops.

### Cross-Validation Strategy
```python
StratifiedKFold(n_splits=3)
```
Ensures each fold maintains the class balance (50/50 positive/negative).

## Conclusion

This notebook demonstrates that recommendation is fundamentally a **ranking problem**, not a classification problem. Perfect classification metrics can be misleading - what matters is whether the system can rank relevant items above irrelevant ones for users.

The poor performance of both models highlights the challenge of recommendation systems: sparse data, implicit feedback, and the need to generalize to unseen items. Real-world systems typically use sophisticated hybrid approaches that combine multiple signals and techniques.

