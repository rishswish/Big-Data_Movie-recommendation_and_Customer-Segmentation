
# 🎬 Scalable Movie Recommendation & Customer Segmentation System

This project explores scalable methods for **movie recommendation** and **customer segmentation** using the [MovieLens dataset](https://grouplens.org/datasets/movielens/latest/). We implement both baseline and advanced models using **Apache Spark** and **HDFS**, enabling big data processing for collaborative filtering and user similarity detection.

---

## 📁 Dataset

We use the **MovieLens** dataset provided by:

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

Two versions of the dataset are available on **Dataproc's HDFS**:

| Dataset        | HDFS Path                                      | Users   | Movies  |
|----------------|-----------------------------------------------|---------|---------|
| Small (prototype) | `/user/pw44_nyu_edu/ml-latest-small.zip`     | 600     | 9,000   |
| Full (scaling) | `/user/pw44_nyu_edu/ml-latest.zip`            | 330,000 | 86,000  |

Each version contains user ratings, tags, and metadata. The full version also includes **tag genome** features, which may enhance the recommendation system.

> ℹ️ **Note:** Use the small dataset for prototyping and the full dataset for final evaluation. Unzip the datasets in HDFS using appropriate shell commands and consult the included `README.txt` files.

---

## ✅ What We Built

### 🔍 Customer Segmentation

1. **Similarity Computation:**
   - Define "movie-watching style" based on the set of movies a user rated (ignoring rating values).
   - Use **MinHash** and **Locality-Sensitive Hashing (LSH)** to find the top 100 most similar user pairs ("movie twins").

2. **Validation:**
   - Compute the average Pearson correlation of ratings within the top 100 user pairs.
   - Compare this to the average correlation in 100 randomly selected user pairs from the full dataset.

---

### 🎥 Movie Recommendation

3. **Data Partitioning:**
   - Split ratings data into **training**, **validation**, and **test** sets.
   - Write a reusable script to create and save these partitions.

4. **Baseline Model:**
   - Implement a **popularity-based recommendation** using average movie ratings or counts.
   - Use this as a performance baseline.

5. **Collaborative Filtering with ALS:**
   - Train a model using **Spark’s ALS** algorithm (`pyspark.ml.recommendation.ALS`).
   - Tune the following hyperparameters:
     - `rank`: number of latent factors
     - `regParam`: regularization parameter
   - Evaluate model performance using **RMSE** and **Precision@K** on validation and test sets.

> 📘 Learn more in the [Spark ALS documentation](https://spark.apache.org/docs/3.0.1/ml-collaborative-filtering.html)

---

## 🛠️ Tools & Technologies

- Apache Spark (PySpark)
- HDFS
- MinHash LSH (Spark MLlib)
- Pandas, NumPy, Matplotlib
- MovieLens Dataset

---


## 📌 Key Takeaways

- **MinHash LSH** offers efficient similarity detection for large user-item matrices.
- **Baseline popularity models** provide a useful comparison for collaborative filtering.
- **ALS** performance depends heavily on hyperparameter tuning and proper data partitioning.


