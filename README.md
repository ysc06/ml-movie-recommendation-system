# MovieLens 1M — Item-Based Collaborative Filtering (IBCF) Recommendation System  
**CS 598 PSL — Practical Statistical Learning (Fall 2025)**

## Project Overview  
This project builds a movie recommendation system using the **MovieLens 1M** dataset (1,000,209 ratings from 6,040 users on 3,900 movies).  
The assignment includes:

1. Recommending the **Top 10 Most Popular Movies** using a self-defined popularity metric.  
2. Implementing **Item-Based Collaborative Filtering (IBCF)** entirely from scratch:
   - Building and normalizing the rating matrix  
   - Computing movie–movie cosine similarity  
   - Keeping only the top 30 similarities per movie  
   - Predicting ratings for new users  
   - Falling back to popularity when predictions are insufficient  

Two users are tested:

- The user in **row 1500** from the rating matrix  
- A hypothetical user who rates  
  - *Star Wars: Episode IV – A New Hope (1977)* = 5  
  - *Independence Day (ID4) (1996)* = 4  

---

## Part A — Defining Movie Popularity  
To avoid highly rated but seldom-rated movies dominating the ranking, popularity is defined using a **Bayesian weighted rating/IMDB-style**:

\[
\text{Popularity Score} = \frac{v}{v+m}R + \frac{m}{v+m}C
\]

Where:

- \(R\): movie’s average rating  
- \(v\): number of ratings  
- \(C\): global average rating  
- \(m\): 25th percentile of all rating counts  

This produces a fair and stable popularity ranking.  
The final Top 10 list includes each movie’s:

- Title  
- Rating count  
- Average rating  
- Weighted popularity score  

---

## Part B — Item-Based Collaborative Filtering (IBCF)

### **1. Rating Matrix Construction & Normalization**  
A user–movie matrix \(R\) is created using pivoting.  
Each user’s ratings are **centered** by subtracting their own mean (computed using non-NA values only).

### **2. Cosine Similarity Computation**  
Cosine similarity between movies \(i\) and \(j\):

\[
\text{cos}(i,j) = \frac{x_i \cdot x_j}{\|x_i\|\|x_j\|}
\]

Rules applied:

- Ignore pairs with **fewer than 3 co-rated users**  
- If denominator = 0 → similarity = NA  
- Similarity is mapped into \([0,1]\) via \((1+\text{cos})/2\)

### **3. Keeping Top 30 Similarities**
For each movie, the **top 30 non-NA similarity values** are retained.  
All others are set to NA.

A similarity submatrix (for the 5 required movies) is displayed using this pruned matrix.

### **4. Custom IBCF Recommendation Function**
Given a new user rating vector:

- Predict ratings for **unrated** movies using:

\[
\hat r_{ai} = \frac{\sum_{j \in N(i)} s_{ij} \, r_{aj}}
                  {\sum_{j \in N(i)} s_{ij}}
\]

- If fewer than 10 predictions → fallback to popularity ranking from Part A  
- Exclude movies already rated by the user  

### **5. Testing the System**
The recommender is tested on:

1. **User at row 1500** of the matrix  
2. **A hypothetical user** rating:
   - Star Wars IV = 5  
   - Independence Day = 4  

Both produce a **Top 10 Recommended Movies** list including:  
- Predicted rating  
- Popularity score  
- Movie title  

---

## Challenges Encountered  

### **1. Sparsity of the Data**
The user–movie matrix is over 95% missing.  
A naïve double loop for pairwise similarities is too slow.  
→ Solved using **NumPy matrix operations**.

### **2. Correct Handling of Missing Values**
Ensuring NA values do not contaminate cosine similarity required careful masking and filtering (<3 co-raters).

### **3. Cold-Start Issues**
Movies with few ratings have unstable similarity vectors.  
→ Part A popularity metric acts as a reliable fallback.

### **4. Index & ID Alignment**
Movie IDs must align correctly across multiple tables (ratings, metadata, similarity matrix).  
Incorrect dtype (str vs int) can break recommendations.

### **5. Memory & Performance**
Computing ~7.6M pairwise similarities is expensive.  
→ Matrix-based implementation significantly reduces runtime.

---
