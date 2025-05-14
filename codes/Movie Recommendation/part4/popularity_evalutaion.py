from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, row_number, collect_list, broadcast
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics

# Start Spark session
print("Start Spark session")
spark = SparkSession.builder.appName(
    "PopularityBaselineWithEvaluation").getOrCreate()

# Load datasets
print("Loading training, validation, and test sets")
ratings_train = spark.read.csv(
    "hdfs:///user/hb2976_nyu_edu/ratings_train_local.csv", header=True, inferSchema=True)
ratings_val = spark.read.csv(
    "hdfs:///user/hb2976_nyu_edu/ratings_val_local.csv", header=True, inferSchema=True)
ratings_test = spark.read.csv(
    "hdfs:///user/hb2976_nyu_edu/ratings_test_local.csv", header=True, inferSchema=True)

# Select relevant columns
ratings_train = ratings_train.select("userId", "movieId", "rating")
ratings_val = ratings_val.select("userId", "movieId", "rating")
ratings_test = ratings_test.select("userId", "movieId", "rating")

# Step 1: Get top 1000 globally highest-rated movies (with at least 50 ratings)
print("Computing top 1000 globally popular movies")
top_movies = (
    ratings_train.groupBy("movieId")
    .agg(avg("rating").alias("avg_rating"), count("*").alias("rating_count"))
    .filter(col("rating_count") >= 50)
    .orderBy(col("avg_rating").desc(), col("rating_count").desc())
    .limit(1000)
    .select("movieId")
)

# Step 2: Get distinct users from validation
print("Extracting distinct users from validation")
users_val = ratings_val.select("userId").distinct()

# Step 3: Cross join using broadcast
print("Generating candidate user-movie pairs (val)")
user_movie_candidates_val = users_val.crossJoin(broadcast(top_movies))

# Step 4: Filter seen items from training
print("Filtering already seen movies")
seen_train = ratings_train.select("userId", "movieId").distinct()
unseen_val = user_movie_candidates_val.join(
    seen_train, on=["userId", "movieId"], how="left_anti").repartition(200)

# Step 5: Select top 100 per user
print("Selecting top 100 unseen movies per user")
windowSpec = Window.partitionBy("userId").orderBy("movieId")
ranked_val = unseen_val.withColumn("rank", row_number().over(windowSpec))
top_100_val = ranked_val.filter(col("rank") <= 100).drop("rank")

# Step 6: Prepare predicted and actual lists
print("Preparing predicted and actual lists for validation")
predicted_val = top_100_val.groupBy("userId").agg(
    collect_list("movieId").alias("predicted"))
actual_val = ratings_val.groupBy("userId").agg(
    collect_list("movieId").alias("actual"))
joined_val = predicted_val.join(actual_val, on="userId").rdd.map(
    lambda row: (row["predicted"], row["actual"]))

# Step 7: Evaluate validation metrics
print("Evaluating on validation set")
metrics_val = RankingMetrics(joined_val)
print("Validation Precision@100:", metrics_val.precisionAt(100))
print("Validation MAP:", metrics_val.meanAveragePrecision)
print("Validation NDCG@100:", metrics_val.ndcgAt(100))

# ---------- Repeat for test set ----------

print("\nRepeat evaluation for test set")
users_test = ratings_test.select("userId").distinct()
user_movie_candidates_test = users_test.crossJoin(broadcast(top_movies))
unseen_test = user_movie_candidates_test.join(
    seen_train, on=["userId", "movieId"], how="left_anti").repartition(200)

ranked_test = unseen_test.withColumn("rank", row_number().over(windowSpec))
top_100_test = ranked_test.filter(col("rank") <= 100).drop("rank")

predicted_test = top_100_test.groupBy("userId").agg(
    collect_list("movieId").alias("predicted"))
actual_test = ratings_test.groupBy("userId").agg(
    collect_list("movieId").alias("actual"))
joined_test = predicted_test.join(actual_test, on="userId").rdd.map(
    lambda row: (row["predicted"], row["actual"]))

metrics_test = RankingMetrics(joined_test)
print("Test Precision@100:", metrics_test.precisionAt(100))
print("Test MAP:", metrics_test.meanAveragePrecision)
print("Test NDCG@100:", metrics_test.ndcgAt(100))
