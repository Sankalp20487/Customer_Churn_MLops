# glue/transform_job.py

import sys
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer

# ────────────────────────────────────────────────────────────────────────────────
# Glue job args:
#   --JOB_NAME
#   --S3_BUCKET_NAME
#   --TRAIN_INPUT_KEY       (e.g. raw/train.csv)
#   --TEST_INPUT_KEY        (e.g. raw/test.csv)
#   --TRAIN_OUTPUT_PREFIX   (e.g. processed/train_cleaned/)
#   --TEST_OUTPUT_PREFIX    (e.g. processed/test_cleaned/)
# ────────────────────────────────────────────────────────────────────────────────
args = getResolvedOptions(
    sys.argv,
    [
      "JOB_NAME",
      "S3_BUCKET_NAME",
      "TRAIN_INPUT_KEY",
      "TEST_INPUT_KEY",
      "TRAIN_OUTPUT_PREFIX",
      "TEST_OUTPUT_PREFIX",
    ],
)

sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session
job         = Job(glueContext)
job.init(args["JOB_NAME"], args)

bucket = args["S3_BUCKET_NAME"]

# Define the two (input,output) pairs
datasets = [
    (args["TRAIN_INPUT_KEY"],  args["TRAIN_OUTPUT_PREFIX"]),
    (args["TEST_INPUT_KEY"],   args["TEST_OUTPUT_PREFIX"]),
]

for input_key, output_prefix in datasets:
    input_path  = f"s3://{bucket}/{input_key}"
    output_path = f"s3://{bucket}/{output_prefix}"

    print(f"➡️  Reading  {input_path}")
    df = (
      spark.read
           .option("header", "true")
           .option("inferSchema", "true")
           .csv(input_path)
    )

    # normalize column names
    for c in df.columns:
        df = df.withColumnRenamed(c, c.strip().lower().replace(" ", "_"))

    if "customerid" in df.columns:
        df = df.drop("customerid")

    df_cleaned = df.dropna().dropDuplicates()

    # index gender
    df_cleaned = (
      StringIndexer(inputCol="gender", outputCol="gender_index")
      .fit(df_cleaned)
      .transform(df_cleaned)
    )

    # map subscription_type
    df_cleaned = df_cleaned.withColumn(
        "subscription_type_index",
        when(col("subscription_type") == "Basic",   0)
        .when(col("subscription_type") == "Standard",1)
        .when(col("subscription_type") == "Premium", 2)
        .otherwise(None)
    )

    # map contract_length
    df_cleaned = df_cleaned.withColumn(
        "contract_length_index",
        when(col("contract_length") == "Monthly",   0)
        .when(col("contract_length") == "Quarterly",1)
        .when(col("contract_length") == "Annual",   2)
        .otherwise(None)
    )

    print(f"➡️  Writing cleaned data to  {output_path}")
    (
      df_cleaned
      .write
      .mode("overwrite")      # drop anything that was there
      .parquet(output_path)
    )

job.commit()
