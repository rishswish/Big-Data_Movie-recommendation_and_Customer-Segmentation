from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as sum_, count, collect_list, row_number, size
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.functions import broadcast
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import expr
# Start Spark session
print("Starting Spark session...")
spark = SparkSession.builder.appName(
    "PopularityBaselineWithBias").getOrCreate()

# Load training and validation data
print("Loading training and validation data...")
train = spark.read.csv("hdfs:///user/mu2253_nyu_edu/ratings_train.csv", header=True,
                       inferSchema=True).select("userId", "movieId", "rating")
val = spark.read.csv("hdfs:///user/mu2253_nyu_edu/ratings_val.csv", header=True,
                     inferSchema=True).select("userId", "movieId", "rating")

# Prepare ground truth from validation set
print("Preparing ground truth...")
val_truth = val.groupBy("userId").agg(
    collect_list("movieId").alias("trueItems"))

# Bias values to try
# bias_values = [0.1, 1, 10, 100, 1000]
# bias_values = [1, 10, 100, 500, 1000, 10000]
bias_values = [1000]
results = []

for b in bias_values:
    print(f"\nTrying b = {b}...")

    print("Computing popularity scores with bias in denominator...")
    top_movies = (
        train.groupBy("movieId")
        .agg(sum_("rating").alias("total_rating"), count("*").alias("rating_count"))
        .withColumn("popularity_score", col("total_rating") / (col("rating_count") + b))
        .orderBy(col("popularity_score").desc())
        .limit(1000)
    )

    users = val.select("userId").distinct()
    print("Generating candidate pairs...")
    # candidates = users.crossJoin(top_movies.select("movieId"))
    candidates = users.crossJoin(broadcast(top_movies))

    seen = train.select("userId", "movieId").distinct()
    print("Filtering out already seen movies...")
    unseen_candidates = candidates.join(
        seen, on=["userId", "movieId"], how="left_anti")
    unseen_candidates = unseen_candidates.repartition(200, "userId")

    print("Selecting top 100 recommendations per user...")
    windowSpec = Window.partitionBy("userId").orderBy("movieId")
    ranked = unseen_candidates.withColumn(
        "rank", row_number().over(windowSpec))
    top_100 = ranked.filter(col("rank") <= 100).drop("rank")

    print("Aggregating predictions...")
    predictions = top_100.withColumn("movieId", col("movieId").cast(DoubleType())) \
        .groupBy("userId").agg(collect_list("movieId").alias("predItems"))

    val_truth = val.withColumn("movieId", col("movieId").cast(DoubleType())) \
        .groupBy("userId").agg(collect_list("movieId").alias("trueItems"))

    joined = predictions.join(
        val_truth, on="userId").filter(size("trueItems") > 0)

    print("Evaluating MAP score...")
    evaluator = RankingEvaluator(
        predictionCol="predItems", labelCol="trueItems", metricName="meanAveragePrecision")
    map_score = evaluator.evaluate(joined)

    print(f"MAP for b={b}: {map_score}")
    results.append((b, map_score))

# Final Results
print("\nAll bias values and MAP scores:")
for b, score in results:
    print(f"b = {b}, MAP = {score}")

best_b, best_score = max(results, key=lambda x: x[1])
print(f"\nBest bias: {best_b}, with MAP = {best_score}")
