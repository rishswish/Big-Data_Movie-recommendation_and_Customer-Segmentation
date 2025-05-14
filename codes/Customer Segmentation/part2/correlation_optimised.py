from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
import pandas as pd
import numpy as np
import random

# --- Start Spark session ---
print("Starting Spark session...")
spark = SparkSession.builder.appName("UserCorrelationComparison").getOrCreate()

# --- Load ratings ---
print("Loading ratings.csv from HDFS...")
ratings_df = spark.read.csv("hdfs:///user/hb2976_nyu_edu/ml-latest/ratings.csv", header=True, inferSchema=True)
print(f"Total ratings loaded: {ratings_df.count()}")

# --- Filter users with at least 50 ratings ---
print("Filtering users with at least 50 ratings...")
user_counts = ratings_df.groupBy("userId").agg(count("*").alias("rating_count"))
active_users = user_counts.filter(col("rating_count") >= 50)
ratings_df = ratings_df.join(active_users, on="userId", how="inner")
print(f"Ratings after filtering: {ratings_df.count()}")

# --- Load Top 100 pairs ---
print("Loading top_100_pairs.csv...")
top_df = pd.read_csv("top_100_pairs.csv")
top_pairs = list(zip(top_df["User1"], top_df["User2"]))
top_user_ids = set(top_df["User1"]).union(set(top_df["User2"]))
print(f"Top user pairs loaded: {len(top_pairs)} with {len(top_user_ids)} unique users")

# --- Get ratings of top 100 users ---
print("Filtering ratings for top 100 users...")
top_ratings_df = ratings_df.filter(col("userId").isin(list(top_user_ids)))
top_user_ratings = top_ratings_df.toPandas()
print("Building user-rating dictionary for top users...")
top_ratings_dict = {
    user_id: dict(zip(group["movieId"], group["rating"]))
    for user_id, group in top_user_ratings.groupby("userId")
}

# --- Compute correlation for top pairs ---
print("Computing average correlation for top 100 pairs...")
top_corrs = []
for idx, (u1, u2) in enumerate(top_pairs):
    print(f"Top pair {idx+1}/100")
    r1, r2 = top_ratings_dict.get(u1, {}), top_ratings_dict.get(u2, {})
    common = set(r1) & set(r2)
    if len(common) > 1:
        v1, v2 = [r1[m] for m in common], [r2[m] for m in common]
        corr = np.corrcoef(v1, v2)[0, 1]
        if not np.isnan(corr):
            top_corrs.append(corr)
print(f"Average correlation (top 100 pairs): {np.mean(top_corrs):.4f}")

# --- Sample 200 users to form 100 random pairs ---
print("Sampling 5000 random users once for all 50 runs...")
sampled_users = ratings_df.select("userId").distinct().rdd.map(lambda r: r["userId"]).takeSample(False, 5000)

print("\nRepeating random 100-pair correlation process 50 times...")
all_avg_corrs = []
for run in range(1,50):
    print(f"\n--- Run {run}/50 ---")

    rand_pairs = []
    while len(rand_pairs) < 100:
        u1, u2 = random.sample(sampled_users, 2)
        if u1 != u2:
            rand_pairs.append((u1, u2))
    rand_user_ids = set(u for p in rand_pairs for u in p)

    # --- Get ratings of random users ---
    rand_ratings_df = ratings_df.filter(col("userId").isin(list(rand_user_ids)))
    rand_user_ratings = rand_ratings_df.toPandas()
    rand_ratings_dict = {
        user_id: dict(zip(group["movieId"], group["rating"]))
        for user_id, group in rand_user_ratings.groupby("userId")
    }

    # --- Compute correlation for random pairs ---

    rand_corrs = []
    for idx, (u1, u2) in enumerate(rand_pairs):
        # print(f"Random pair {idx+1}/100")
        r1, r2 = rand_ratings_dict.get(u1, {}), rand_ratings_dict.get(u2, {})
        common = set(r1) & set(r2)
        if len(common) > 1:
            v1, v2 = [r1[m] for m in common], [r2[m] for m in common]
            corr = np.corrcoef(v1, v2)[0, 1]
            if not np.isnan(corr):
                rand_corrs.append(corr)
    # print(f"Average correlation (random 100 pairs): {np.mean(rand_corrs):.4f}")

    avg_corr = np.mean(rand_corrs) if rand_corrs else float('nan')
    print(f"Average correlation (run {run}): {avg_corr:.4f}")
    all_avg_corrs.append(avg_corr)

# --- Final result ---
final_avg_corr = np.mean([c for c in all_avg_corrs if not np.isnan(c)])
print(f"\nFinal average correlation across 50 runs: {final_avg_corr:.4f}")

# --- Stop Spark ---
spark.stop()
