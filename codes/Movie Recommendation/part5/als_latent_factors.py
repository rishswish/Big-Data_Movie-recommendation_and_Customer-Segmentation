from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, collect_list
from pyspark.sql.types import IntegerType, FloatType
from pyspark.mllib.evaluation import RankingMetrics

# Initialize Spark
print("Initializing Spark session")
spark = SparkSession.builder.appName("ALSModelTuning").getOrCreate()

# Load and preprocess data
print("Loading and preprocessing training, validation, and test data")
def load_and_clean(path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.dropna(subset=["userId", "movieId", "rating"]) \
           .withColumn("userId", col("userId").cast(IntegerType())) \
           .withColumn("movieId", col("movieId").cast(IntegerType())) \
           .withColumn("rating", col("rating").cast(FloatType()))
    return df.select("userId", "movieId", "rating")

train = load_and_clean("hdfs:///user/mu2253_nyu_edu/ratings_train.csv")
val = load_and_clean("hdfs:///user/mu2253_nyu_edu/ratings_val.csv")
test = load_and_clean("hdfs:///user/mu2253_nyu_edu/ratings_test.csv")

# Prepare ground truth for RankingMetrics
print("Preparing ground truth")
val_truth = val.groupBy("userId").agg(collect_list("movieId").alias("actual"))
test_truth = test.groupBy("userId").agg(collect_list("movieId").alias("actual"))

# Hyperparameter tuning
ranks = [5,10,20,50]
regParams = [0.01,0.05, 0.1, 1]
best_model = None
best_map = -1
best_rank = None
best_reg = None

for rank in ranks:
    for reg in regParams:
        print(f"Training ALS model with rank={rank}, regParam={reg}")
        als = ALS(
            userCol="userId", itemCol="movieId", ratingCol="rating",
            rank=rank, regParam=reg, implicitPrefs=False,
            coldStartStrategy="drop", nonnegative=True
        )
        model = als.fit(train)

        # Generate top-100 recommendations on validation set
        print("Generating recommendations on validation set")
        val_users = val.select("userId").distinct()
        recs = model.recommendForUserSubset(val_users, 100)

        # Format predictions for RankingMetrics
        pred = recs.select("userId", "recommendations.movieId") \
                   .withColumnRenamed("movieId", "predicted")
        pred = pred.join(val_truth, on="userId").rdd.map(
            lambda row: (row["predicted"], row["actual"])
        )
        metrics = RankingMetrics(pred)
        map_score = metrics.meanAveragePrecision
        print("\nVal Precision@100:", metrics.precisionAt(100))
        print("Val MAP:", map_score)
        print("Val NDCG@100:", metrics.ndcgAt(100))

        # Update best model
        if map_score > best_map:
            best_model = model
            best_map = map_score
            best_rank = rank
            best_reg = reg

print(f"\nBest model: rank={best_rank}, regParam={best_reg}, MAP={best_map}")

# Evaluate fixed model (rank=50, reg=0.05) on test set
print("Evaluating fixed model (rank=50, reg=0.05) on test set")
fixed_model = ALS(
    userCol="userId", itemCol="movieId", ratingCol="rating",
    rank=50, regParam=0.05, implicitPrefs=False,
    coldStartStrategy="drop", nonnegative=True
).fit(train)

# Filter test users to only those in training set
test_users = test.select("userId").distinct()
test_users = test_users.join(train.select("userId").distinct(), on="userId")
test_recs = fixed_model.recommendForUserSubset(test_users, 100)

test_pred = test_recs.select("userId", "recommendations.movieId") \
    .withColumnRenamed("movieId", "predicted")
test_pred = test_pred.join(test_truth, on="userId").rdd.map(
    lambda row: (row["predicted"], row["actual"])
)
test_metrics = RankingMetrics(test_pred)

# Compute RMSE on actual test ratings
test_predictions = fixed_model.transform(test)
rmse_evaluator = RegressionEvaluator(
    metricName="rmse", labelCol="rating", predictionCol="prediction"
)
rmse = rmse_evaluator.evaluate(test_predictions)

# Output test metrics
print("\nTest Precision@100:", test_metrics.precisionAt(100))
print("Test MAP:", test_metrics.meanAveragePrecision)
print("Test NDCG@100:", test_metrics.ndcgAt(100))
print("Test RMSE:", rmse)

print("Done.")
