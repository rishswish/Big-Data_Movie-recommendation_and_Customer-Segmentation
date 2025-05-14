from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, row_number, floor
from pyspark.sql.window import Window

# --- STEP 1: Start Spark session ---
print("Starting Spark session...")
spark = SparkSession.builder.appName("PartitionRatings").getOrCreate()

# --- STEP 2: Load the dataset ---
print("Loading ratings.csv...")
df = spark.read.csv("hdfs:///user/rbp5812_nyu_edu/ml-latest/ratings.csv", header=True, inferSchema=True)
print(f"Total records loaded: {df.count()}")

# --- STEP 3: Filter users with at least 5 ratings ---
print("Filtering users with at least 5 ratings...")
user_counts = df.groupBy("userId").agg(count("*").alias("num_ratings"))
active_users = user_counts.filter(col("num_ratings") >= 5)
df_filtered = df.join(active_users, on="userId", how="inner")
print(f"Remaining ratings after filter: {df_filtered.count()}")

# --- STEP 4: Add row numbers per user ---
print("Adding row numbers per user...")
window = Window.partitionBy("userId").orderBy("timestamp")
df_with_row = df_filtered.withColumn("row_num", row_number().over(window))

# --- STEP 5: Disambiguate num_ratings column before calculating splits ---
df_cleaned = df_with_row.select(
    "userId", "movieId", "rating", "timestamp", "row_num", active_users["num_ratings"]
)

print("Assigning split labels...")
df_partitioned = df_cleaned.withColumn(
    "split",
    (floor((col("row_num") - 1) / col("num_ratings") * 5)).cast("int")
)

# --- STEP 6: Write split data to HDFS ---
print("Saving training set...")
df_partitioned.filter(col("split") < 3) \
    .select("userId", "movieId", "rating", "timestamp") \
    .write.mode("overwrite").csv("ratings_train", header=True)

print("Saving validation set...")
df_partitioned.filter((col("split") == 3)) \
    .select("userId", "movieId", "rating", "timestamp") \
    .write.mode("overwrite").csv("ratings_val", header=True)

print("Saving test set...")
df_partitioned.filter((col("split") == 4)) \
    .select("userId", "movieId", "rating", "timestamp") \
    .write.mode("overwrite").csv("ratings_test", header=True)

print("Done writing all splits.")
