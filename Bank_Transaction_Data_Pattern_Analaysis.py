# Databricks notebook source
# MAGIC %md
# MAGIC ## Load Transactions to S3 
# MAGIC

# COMMAND ----------

# spark.conf.set("spark.databricks.env.AWS_ACCESS_KEY", "your-access-key")
# spark.conf.set("spark.databricks.env.AWS_SECRET_KEY", "your-secret-key")
# spark.conf.set("spark.databricks.env.S3_BUCKET_NAME", "your-bucket-name")
# spark.conf.set("spark.databricks.env.GDRIVE_FILE_ID", "your-gdrive-id")


# COMMAND ----------

import pandas as pd
import boto3
import io
import os
from datetime import datetime
from pyspark.sql import SparkSession
from delta.tables import DeltaTable
from pyspark.sql.utils import AnalysisException

# Spark Session
spark = SparkSession.builder \
    .appName("LoadTransactions") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

AWS_ACCESS_KEY = spark.conf.get("spark.databricks.env.AWS_ACCESS_KEY")
AWS_SECRET_KEY = spark.conf.get("spark.databricks.env.AWS_SECRET_KEY")
S3_BUCKET_NAME = spark.conf.get("spark.databricks.env.S3_BUCKET_NAME")
GDRIVE_FILE_ID = spark.conf.get("spark.databricks.env.GDRIVE_FILE_ID")


# S3 Client
s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# GDrive URL
file_id = GDRIVE_FILE_ID.split("/")[-2]
download_url = f"https://drive.google.com/uc?id={file_id}"


BATCH_SIZE = 10000


CHECKPOINT_TABLE = "default.transactions_data_checkpoint"

try:
    spark.table(CHECKPOINT_TABLE)  # Try accessing the table
    print(f"Table '{CHECKPOINT_TABLE}' already exists.")
except AnalysisException:
    print(f"Table '{CHECKPOINT_TABLE}' does not exist. Creating now...")

    # Create the Delta table with one row containing 0
    spark.sql(f"""
        CREATE TABLE  {CHECKPOINT_TABLE} (
            id INT
        ) USING DELTA
        LOCATION 'dbfs:/user/hive/warehouse/transactions_data_checkpoint'
    """)

    # Insert the initial value
    spark.sql(f"INSERT INTO {CHECKPOINT_TABLE} VALUES (0)")


# Process Chunk
def process_next_chunk():
    while True:
        df = pd.read_csv(download_url)
        spark_df = spark.createDataFrame(df)

        last_idx = spark.sql(f"SELECT max(id) FROM {CHECKPOINT_TABLE}").collect()[0][0]
        chunk = df.iloc[last_idx:last_idx + BATCH_SIZE]
        if chunk.empty:
            print("No new data.")
            return

        # Upload chunk to S3
        csv_buf = io.StringIO()
        chunk.to_csv(csv_buf, index=False)
        file_key = f"transaction_data/transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        s3.put_object(Bucket=S3_BUCKET_NAME, Key=file_key, Body=csv_buf.getvalue())

        # Update checkpoint
        spark.sql(f"DELETE FROM {CHECKPOINT_TABLE}")
        spark.sql(f"INSERT INTO {CHECKPOINT_TABLE} VALUES ({last_idx + len(chunk)})")
        print(f"Uploaded {len(chunk)} records to {file_key}") 
process_next_chunk()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Customer Pattern Detection 

# COMMAND ----------

import os
import time
import io
from datetime import datetime, timedelta
import pytz
import boto3
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, lit, when
from pyspark.sql.functions import percentile_approx

# ---------- SETUP ----------

# Load AWS credentials from env
AWS_ACCESS_KEY = spark.conf.get("spark.databricks.env.AWS_ACCESS_KEY")
AWS_SECRET_KEY = spark.conf.get("spark.databricks.env.AWS_SECRET_KEY")
S3_BUCKET= spark.conf.get("spark.databricks.env.S3_BUCKET_NAME")
# IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Spark Session
spark = SparkSession.builder \
    .appName("MechanismY_PatternDetection") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

spark.conf.set("fs.s3a.access.key", AWS_ACCESS_KEY)
spark.conf.set("fs.s3a.secret.key", AWS_SECRET_KEY)
spark.conf.set("fs.s3a.endpoint", "s3.amazonaws.com")

# AWS S3 boto3 client
s3_client = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# Paths
TRANSACTION_PATH = f"s3a://{S3_BUCKET}/transaction_data/"
IMPORTANCE_PATH = f"s3a://{S3_BUCKET}/customer_importance/"
OUTPUT_PATH = f"s3a://{S3_BUCKET}/detections/"

# ---------- READ DATA ----------

transactions_df = spark.read.option("header", "true").csv(TRANSACTION_PATH)
transactions_df = transactions_df.withColumn("amount", col("amount").cast("double"))

importance_df = spark.read.option("header", "true").csv(IMPORTANCE_PATH)
importance_df = importance_df.withColumn("Weight", col("Weight").cast("double"))

# ---------- MERGE DATA ----------

merged_df = transactions_df.join(importance_df,
    (transactions_df["customer"] == importance_df["Source"]) &
    (transactions_df["merchant"] == importance_df["Target"]), "left"
).select(transactions_df["*"], importance_df["Weight"])


now_ist = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')

# ---------- PATTERN 1: ----------

agg_df = merged_df.groupBy("merchant", "customer") \
    .agg(count("*").alias("total_txns"), avg("Weight").alias("avg_weight"))

merchant_txn_counts = merged_df.groupBy("merchant") \
    .agg(count("*").alias("merchant_txns"))

percentiles_df = agg_df.groupBy("merchant").agg(
    percentile_approx("total_txns", 0.99).alias("txn_99"),
    percentile_approx("avg_weight", 0.01).alias("weight_01")
)

pat1_df = agg_df.join(percentiles_df, "merchant") \
    .join(merchant_txn_counts, "merchant") \
    .filter(
        (col("merchant_txns") > 50000) &
        (col("total_txns") >= col("txn_99")) &
        (col("avg_weight") <= col("weight_01"))
    ).selectExpr(
        f"'{now_ist}' as YStartTime",
        f"'{now_ist}' as detectionTime",
        "'PatId1' as patternId",
        "'UPGRADE' as ActionType",
        "customer as CustomerName",
        "merchant as MerchantId"
    )

# ---------- PATTERN 2: CHILD ----------

pat2_df = merged_df.groupBy("merchant", "customer") \
    .agg(avg("amount").alias("avg_amt"), count("*").alias("total_txns")) \
    .filter((col("avg_amt") < 23) & (col("total_txns") >= 80)) \
    .selectExpr(
        f"'{now_ist}' as YStartTime",
        f"'{now_ist}' as detectionTime",
        "'PatId2' as patternId",
        "'CHILD' as ActionType",
        "customer as CustomerName",
        "merchant as MerchantId"
    )

# ---------- PATTERN 3: DEI-NEEDED ----------

gender_stats = transactions_df.groupBy("merchant", "gender") \
    .agg(count("customer").alias("gender_count"))

# Pivot to get male/female counts per merchant
gender_pivot = gender_stats.groupBy("merchant").pivot("gender", ["Male", "Female"]).sum("gender_count") \
    .na.fill(0)

pat3_df = gender_pivot.filter(
    (col("Female") > 100) & (col("Female") < col("Male"))
).selectExpr(
    f"'{now_ist}' as YStartTime",
    f"'{now_ist}' as detectionTime",
    "'PatId3' as patternId",
    "'DEI-NEEDED' as ActionType",
    "'' as CustomerName",
    "merchant as MerchantId"
)

# ---------- COMBINE & BATCH ----------

all_detections = pat1_df.union(pat2_df).union(pat3_df)
detection_rows = all_detections.collect()

batch_size = 50
for i in range(0, len(detection_rows), batch_size):
    batch = detection_rows[i:i+batch_size]
    batch_df = spark.createDataFrame(batch)

    # Convert to CSV
    csv_buffer = io.StringIO()
    batch_df.toPandas().to_csv(csv_buffer, index=False)

    filename = f"detection_{int(time.time())}_{i//batch_size}.csv"
    s3_client.put_object(Bucket=S3_BUCKET, Key=f"detections/{filename}", Body=csv_buffer.getvalue())
    # print(f"Uploaded: {filename} [{len(batch)} records]")

print("Pattern detection completed.")
