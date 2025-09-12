# Deep Dive: Recommendation System Metrics and ALS Mathematics
## A Step-by-Step Explanation

## Part 1: Classification Metrics

### 1.1 ROC-AUC Score (Receiver Operating Characteristic - Area Under Curve)

#### What It Is
ROC-AUC measures how well a model can distinguish between positive and negative classes across all possible classification thresholds.

#### How It Works

**Step 1: Understanding Predictions**
```python
# Model outputs probabilities, not binary decisions
scores = [0.9, 0.7, 0.6, 0.3, 0.1]  # Predicted probabilities
labels = [1,   1,   0,   1,   0]     # Actual labels
```

**Step 2: The ROC Curve**
The ROC curve plots:
- **Y-axis:** True Positive Rate (TPR) = Sensitivity = TP/(TP+FN)
- **X-axis:** False Positive Rate (FPR) = 1-Specificity = FP/(FP+TN)

For each possible threshold (0.0 to 1.0):
```
Threshold = 0.5:
Predictions: [1, 1, 1, 0, 0]  # scores > 0.5 → predict 1
Actual:      [1, 1, 0, 1, 0]
TPR = 2/3 = 0.67  (found 2 out of 3 positives)
FPR = 1/2 = 0.50  (misclassified 1 out of 2 negatives)
```

**Step 3: Calculating AUC**
```python
# AUC is literally the area under the ROC curve
# Computed using trapezoidal rule integration
roc_auc_score(labels, scores)
```

**Interpretation:**
- **AUC = 1.0:** Perfect separator - model ranks all positives above all negatives
- **AUC = 0.5:** Random guessing - no discrimination ability
- **AUC = 0.0:** Perfect but inverted - all predictions are backwards

**Example Calculation:**
```python
# Simplified example
scores = [0.9, 0.8, 0.7, 0.6, 0.5]
labels = [1,   1,   0,   0,   1]

# Rank by scores (descending):
# Score: 0.9  0.8  0.7  0.6  0.5
# Label: 1    1    0    0    1
# 
# As we move threshold from 1.0 to 0.0:
# We encounter: positive, positive, negative, negative, positive
# This creates the ROC curve points
```

#### Why It Matters for Recommendations
- Measures if the system ranks items users like higher than items they don't
- Robust to class imbalance (works even if 90% of items are irrelevant)
- Single number summary of ranking quality

### 1.2 Average Precision Score (AP or PR-AUC)

#### What It Is
Average Precision focuses on the precision-recall trade-off, particularly important when positive class is rare.

#### How It Works

**The Precision-Recall Curve:**
```python
# For each threshold, calculate:
Precision = TP / (TP + FP)  # Of items we said "yes", how many were correct?
Recall = TP / (TP + FN)     # Of all "yes" items, how many did we find?
```

**Average Precision Calculation:**
```python
average_precision_score(labels, scores)
```

This computes:
```
AP = Σₙ (Rₙ - Rₙ₋₁) × Pₙ
```

Where:
- Rₙ = Recall at nth threshold
- Pₙ = Precision at nth threshold

**Step-by-step Example:**
```python
scores = [0.9, 0.8, 0.35, 0.4]
labels = [1,   0,   1,    0]

# Sort by score (descending):
# Position 1: score=0.9, label=1 → Precision=1/1=1.0, Recall=1/2=0.5
# Position 2: score=0.8, label=0 → Precision=1/2=0.5, Recall=1/2=0.5
# Position 3: score=0.4, label=0 → Precision=1/3=0.33, Recall=1/2=0.5
# Position 4: score=0.35, label=1 → Precision=2/4=0.5, Recall=2/2=1.0

# AP = weighted average of precisions at each relevant item
AP = (1.0 × 0.5) + (0.5 × 0.5) = 0.75
```

#### Why AP vs ROC-AUC?
- **AP** is better when positive class is rare (most recommendations are irrelevant)
- **ROC-AUC** treats both classes equally
- In recommendations, we care more about finding relevant items (positive class)

## Part 2: Ranking Metrics

### 2.1 Precision@k

#### The Formula
```
Precision@k = |relevant items in top-k| / k
```

#### Detailed Explanation

**What It Measures:**
Of the top-k items we recommend, what fraction are actually relevant?

**Step-by-step Calculation:**
```python
# User's actual preferences (1 = likes, 0 = doesn't like)
all_items = [0, 1, 0, 1, 1, 0, 0, 1, 0, 0]

# Our predicted ranking (sorted by score)
top_10_predictions = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]

# Precision@5
top_5 = top_10_predictions[:5]  # [1, 1, 0, 0, 1]
relevant_in_top_5 = sum(top_5)  # 3
precision_at_5 = 3 / 5 = 0.6
```

**Why k Matters:**
- **Small k (e.g., 5):** Focus on very top recommendations (what user sees first)
- **Large k (e.g., 100):** Overall quality of recommendation list

### 2.2 Recall@k

#### The Formula
```
Recall@k = |relevant items in top-k| / |all relevant items|
```

#### Detailed Explanation

**What It Measures:**
Of all items the user would like, what fraction did we include in our top-k?

**Step-by-step Calculation:**
```python
# Total relevant items for user
total_relevant = sum(all_items)  # 4 items user likes

# Using same top-5 from before
relevant_in_top_5 = 3
recall_at_5 = 3 / 4 = 0.75

# We found 75% of items user likes in our top-5
```

**Precision vs Recall Trade-off:**
- High Precision: "Be conservative, only recommend sure things"
- High Recall: "Be comprehensive, don't miss anything user might like"

### 2.3 NDCG@k (Normalized Discounted Cumulative Gain)

#### The Complete Formula Breakdown

**Step 1: Gain**
```python
gain = 2^relevance - 1
```
- For binary relevance (0 or 1):
  - Irrelevant item (rel=0): gain = 2^0 - 1 = 0
  - Relevant item (rel=1): gain = 2^1 - 1 = 1

**Step 2: Discount Factor**
```python
discount = 1 / log₂(position + 1)
```
- Position 1: 1/log₂(2) = 1.000
- Position 2: 1/log₂(3) = 0.631
- Position 3: 1/log₂(4) = 0.500
- Position 10: 1/log₂(11) = 0.289

**Step 3: Discounted Gain**
```python
discounted_gain = gain × discount = (2^rel - 1) / log₂(pos + 1)
```

**Step 4: DCG (Discounted Cumulative Gain)**
```python
DCG@k = Σᵢ₌₁ᵏ (2^relᵢ - 1) / log₂(i + 1)
```

**Full Example:**
```python
# Predicted ranking with relevance scores
predictions = [1, 1, 0, 1, 0]  # top-5 items
positions =   [1, 2, 3, 4, 5]

# Calculate DCG@5
dcg = 0
dcg += (2^1 - 1) / log₂(1+1) = 1.0 / 1.0 = 1.000  # position 1
dcg += (2^1 - 1) / log₂(2+1) = 1.0 / 1.585 = 0.631  # position 2  
dcg += (2^0 - 1) / log₂(3+1) = 0.0 / 2.0 = 0.000  # position 3
dcg += (2^1 - 1) / log₂(4+1) = 1.0 / 2.322 = 0.431  # position 4
dcg += (2^0 - 1) / log₂(5+1) = 0.0 / 2.585 = 0.000  # position 5
# DCG@5 = 2.062
```

**Step 5: IDCG (Ideal DCG)**
The DCG of the perfect ranking (all relevant items first):
```python
ideal_ranking = [1, 1, 1, 0, 0]  # 3 relevant items placed first
idcg = 1.0 + 0.631 + 0.5 = 2.131
```

**Step 6: Normalization**
```python
NDCG@5 = DCG@5 / IDCG@5 = 2.062 / 2.131 = 0.968
```

#### Why NDCG Is Powerful
1. **Position-aware:** Items at top positions matter more (logarithmic decay)
2. **Normalized:** Comparable across users with different numbers of relevant items
3. **Graded relevance:** Can handle non-binary relevance (ratings 1-5)

## Part 3: ALS (Alternating Least Squares) Deep Dive

### 3.1 Matrix Factorization: The Core Concept

#### The Fundamental Equation
```
R ≈ U × V^T
```

#### Visual Understanding
```
Original Matrix R (5000 users × 400 items):
[r₁₁  r₁₂  ...  r₁,₄₀₀]
[r₂₁  r₂₂  ...  r₂,₄₀₀]
[...  ...  ...  ...]
[r₅₀₀₀,₁  ...  r₅₀₀₀,₄₀₀]

Factorizes into:

U (5000 × 64)        V^T (64 × 400)
[u₁₁ ... u₁,₆₄]     [v₁₁ ... v₁,₄₀₀]
[u₂₁ ... u₂,₆₄]  ×  [v₂₁ ... v₂,₄₀₀]
[... ... ...]       [... ... ...]
[u₅₀₀₀,₁...u₅₀₀₀,₆₄] [v₆₄,₁...v₆₄,₄₀₀]
```

#### What Are Latent Factors?

The 64 latent factors might represent abstract concepts:
- Factor 1: "Likes action content"
- Factor 2: "Prefers social interaction"
- Factor 3: "Evening user"
- Factor 4-64: Other patterns in the data

**User vector example:**
```python
user_123 = [0.8, 0.2, 0.1, ...]  # High on factor 1 (action), low on others
```

**Item vector example:**
```python
game_456 = [0.9, 0.1, 0.0, ...]  # High action content, low social
```

**Prediction:**
```python
score = np.dot(user_123, game_456)  # High score due to matching action preference
```

### 3.2 The Optimization Problem: Complete Breakdown

#### The Loss Function
```
L = Σᵤ,ᵢ cᵤᵢ(rᵤᵢ - uᵤᵀvᵢ)² + λ(||uᵤ||² + ||vᵢ||²)
```

#### Term by Term Explanation

**1. The Prediction Error:**
```
(rᵤᵢ - uᵤᵀvᵢ)²
```
- rᵤᵢ = actual interaction (1 if user u interacted with item i, 0 otherwise)
- uᵤᵀvᵢ = predicted interaction (dot product of user and item vectors)
- Squared difference penalizes larger errors more

**2. The Confidence Weight:**
```
cᵤᵢ = 1 + α × rᵤᵢ
```
Typical values:
- If no interaction (rᵤᵢ = 0): cᵤᵢ = 1 (baseline confidence)
- If interaction exists (rᵤᵢ > 0): cᵤᵢ = 1 + 40×rᵤᵢ (high confidence)

This means:
- We're somewhat confident non-interactions mean "not interested"
- We're very confident interactions mean "interested"

**3. The Regularization Terms:**
```
λ(||uᵤ||² + ||vᵢ||²)
```

Where ||uᵤ||² = Σₖ u²ᵤₖ (sum of squared elements)

**Why regularization?**
- Prevents overfitting by keeping vector values small
- Without it, model might memorize training data perfectly but fail on new data
- λ controls the trade-off (typical values: 0.01 to 1.0)

#### Complete Example with Numbers

```python
# Single user-item pair
user_vector = [0.5, 0.3]  # 2D for simplicity
item_vector = [0.4, 0.6]
actual_interaction = 1
confidence = 40  # High confidence for positive interaction
lambda_reg = 0.1

# Prediction
prediction = 0.5×0.4 + 0.3×0.6 = 0.2 + 0.18 = 0.38

# Error term
error = (1 - 0.38)² = 0.3844

# Weighted error
weighted_error = 40 × 0.3844 = 15.376

# Regularization
reg_user = 0.1 × (0.5² + 0.3²) = 0.1 × 0.34 = 0.034
reg_item = 0.1 × (0.4² + 0.6²) = 0.1 × 0.52 = 0.052

# Total loss for this pair
loss = 15.376 + 0.034 + 0.052 = 15.462
```

### 3.3 The "Alternating" Algorithm

#### Why Alternating?
The full optimization is non-convex (no single global minimum). But if we fix one set of variables, it becomes convex!

#### The Two Steps

**Step 1: Fix V, Solve for U**
For each user u:
```
uᵤ = (VᵀCᵤV + λI)⁻¹ VᵀCᵤrᵤ
```

Where:
- Cᵤ = diagonal matrix of confidences for user u
- rᵤ = vector of user u's interactions
- I = identity matrix

**Step 2: Fix U, Solve for V**
For each item i:
```
vᵢ = (UᵀCᵢU + λI)⁻¹ UᵀCᵢrᵢ
```

#### Intuition Behind Alternating
1. Start with random U and V
2. Fix V, find best U that minimizes loss
3. Fix new U, find best V that minimizes loss
4. Repeat until convergence

It's like optimizing a 3D surface by always moving along one axis at a time.

### 3.4 Making Predictions

After training, predict user u's score for item i:
```python
score = np.dot(U[u], V[i])
```

For all items at once:
```python
all_scores = U[u] @ V.T  # Vector of scores for all items
top_k = np.argsort(all_scores)[-k:]  # Get top-k items
```

## Summary: Connecting It All

1. **Metrics** measure different aspects of recommendation quality:
   - **Classification metrics** (ROC-AUC, AP): Can we distinguish good from bad?
   - **Ranking metrics** (Precision, Recall, NDCG): Can we put good items at the top?

2. **ALS** learns patterns from the interaction matrix:
   - Decomposes sparse matrix into dense factors
   - Uses confidence weighting for implicit feedback
   - Alternates optimization for tractability

3. **The key insight**: Perfect classification (AUC=1.0) doesn't guarantee good ranking (low Recall@k) because recommendation is fundamentally about ranking, not binary classification.

Each metric and algorithm component serves a specific purpose in building effective recommendation systems. Understanding these details helps diagnose why models succeed or fail on real data.