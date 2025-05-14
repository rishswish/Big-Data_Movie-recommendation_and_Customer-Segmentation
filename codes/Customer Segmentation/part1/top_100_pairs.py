import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, count
from datasketch import MinHash, MinHashLSH
from math import ceil
import pandas as pd
import heapq

# --- Setup logging ---
logging.basicConfig(
    filename="lsh_pipeline.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger()

# --- STEP 1: Start Spark session ---
print("Starting Spark session...")
log.info("Starting Spark session...")
spark = SparkSession.builder.appName("UserSimilarityLSH").getOrCreate()

# --- STEP 2: Load data from HDFS ---
print("Loading dataset...")
log.info("Loading dataset from HDFS...")
df = spark.read.csv("hdfs:///user/mu2253_nyu_edu/ml-latest/ratings.csv", header=True, inferSchema=True)
record_count = df.count()
print(f"Total records loaded: {record_count}")
log.info(f"Total records loaded: {record_count}")

# --- STEP 3: Filter users with at least 50 ratings ---
print("Filtering users with at least 50 ratings...")
log.info("Filtering users with at least 50 ratings...")
user_counts = df.groupBy("userId").agg(count("*").alias("rating_count"))
active_users = user_counts.filter(col("rating_count") >= 50)
df_filtered = df.join(active_users, on="userId", how="inner")
active_user_count = active_users.count()
filtered_count = df_filtered.count()
print(f"Number of active users: {active_user_count}")
print(f"Filtered dataset size: {filtered_count}")
log.info(f"Number of active users: {active_user_count}")
log.info(f"Filtered dataset size: {filtered_count}")

# --- STEP 4: Group movies per user ---
print("Grouping movie IDs for each user...")
log.info("Grouping movie IDs per user...")
user_movies_df = df_filtered.groupBy("userId").agg(collect_set("movieId").alias("movie_ids")).orderBy("userId")
#user_movies_df = user_movies_df.limit(500)
total_users = user_movies_df.count()
print(f"Total users after grouping: {total_users}")
log.info(f"Total users after grouping: {total_users}")

# --- STEP 5: Build LSH in Batches ---
num_perm = 32
batch_size = 5000
num_batches = ceil(total_users / batch_size)

user_signatures = {}
lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)

print("Building LSH index in batches...")
log.info("Building LSH index in batches...")
for i in range(num_batches):
    start = i * batch_size
    end = min(start + batch_size, total_users)
    print(f"Processing users {start} to {end}...")
    log.info(f"Processing batch {i+1}/{num_batches}: users {start} to {end}")

    indexed = user_movies_df.rdd.zipWithIndex().toDF(["row", "idx"])
    batch_df = indexed.filter((col("idx") >= start) & (col("idx") < end)).select("row.*")
    batch = batch_df.collect()

    for row in batch:
        user_id = row["userId"]
        movie_ids = row["movie_ids"]
        m = MinHash(num_perm=num_perm)
        for movie_id in movie_ids:
            m.update(str(movie_id).encode("utf8"))
        user_signatures[user_id] = m
        lsh.insert(str(user_id), m)

# --- STEP 6: Query from global LSH ---
print("Querying similar user pairs...")
log.info("Querying similar user pairs from LSH index...")
candidate_pairs = set()
for u1, m1 in user_signatures.items():
    result = lsh.query(m1)
    for u2 in result:
        u2 = int(u2)
        if u1 != u2:
            pair = tuple(sorted((u1, u2)))
            candidate_pairs.add(pair)
print(f"Total candidate pairs: {len(candidate_pairs)}")
log.info(f"Total candidate pairs found: {len(candidate_pairs)}")

# --- STEP 7: Score pairs and extract top 100 ---
print("Scoring and extracting top 100 pairs...")
log.info("Scoring and extracting top 100 similar user pairs...")
top_100 = []
for u1, u2 in candidate_pairs:
    sim = user_signatures[u1].jaccard(user_signatures[u2])
    if len(top_100) < 100:
        heapq.heappush(top_100, (sim, (u1, u2)))
    else:
        heapq.heappushpop(top_100, (sim, (u1, u2)))

top_100 = sorted(top_100, reverse=True)

top_df = pd.DataFrame([(u1, u2, score) for (score, (u1, u2)) in top_100],
                      columns=["User1", "User2", "Similarity"])
top_df.to_csv("top_100_pairs.csv", index=False)
print("Top 100 pairs written to top_100_pairs.csv")
log.info("Top 100 pairs written to top_100_pairs.csv")

for (score, (u1, u2)) in top_100:
    print(f"Users {u1} and {u2} → Jaccard similarity ≈ {score:.3f}")
    log.info(f"Users {u1} and {u2} → Jaccard similarity ≈ {score:.3f}")

log.info("Job completed successfully.")