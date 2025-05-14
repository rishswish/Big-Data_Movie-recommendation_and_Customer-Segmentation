from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as sum_, row_number, collect_list, broadcast
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics

# Start Spark session
print("Start Spark session")
spark = SparkSession.builder.appName(
    "PopularityBaselineWithBiasAndEvaluation").getOrCreate()

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

# Step 1: Compute popularity score with b = 1000
print("Computing popularity scores with b = 1000")
b = 1000
top_movies = (
    ratings_train.groupBy("movieId")
    .agg(sum_("rating").alias("total_rating"), count("*").alias("rating_count"))
    .withColumn("popularity_score", col("total_rating") / (col("rating_count") + b))
    .orderBy(col("popularity_score").desc())
    .limit(1000)
    .select("movieId")
)

windowSpec = Window.partitionBy("userId").orderBy("movieId")
seen_train = ratings_train.select("userId", "movieId").distinct()

# ---------- Validation Set ----------
print("Evaluating on validation set")
users_val = ratings_val.select("userId").distinct()
user_movie_candidates_val = users_val.crossJoin(broadcast(top_movies))
unseen_val = user_movie_candidates_val.join(
    seen_train, on=["userId", "movieId"], how="left_anti").repartition(200)
ranked_val = unseen_val.withColumn("rank", row_number().over(windowSpec))
top_100_val = ranked_val.filter(col("rank") <= 100).drop("rank")

predicted_val = top_100_val.groupBy("userId").agg(
    collect_list("movieId").alias("predicted"))
actual_val = ratings_val.groupBy("userId").agg(
    collect_list("movieId").alias("actual"))
joined_val = predicted_val.join(actual_val, on="userId").rdd.map(
    lambda row: (row["predicted"], row["actual"]))

metrics_val = RankingMetrics(joined_val)
print("Validation Precision@100:", metrics_val.precisionAt(100))
print("Validation MAP:", metrics_val.meanAveragePrecision)
print("Validation NDCG@100:", metrics_val.ndcgAt(100))

# ---------- Test Set ----------
print("Evaluating on test set")
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
