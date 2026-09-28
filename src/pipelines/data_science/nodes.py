"""
Data science nodes for the Loan Offer Model pipeline.

These nodes transform prepared loan, customer, and transaction data into
consolidated data and engineered features.  Each function takes one or more
Spark DataFrames and returns a transformed Spark DataFrame.  The pipeline.py
module wires these nodes together in sequence using the project's Catalog
class for I/O.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import numpy as np
import pandas as pd
from scipy.stats import skew


# --------------------------------------------------------------------------- #
# Loan feature nodes                                                           #
# --------------------------------------------------------------------------- #

def create_loan_features(loans_df: DataFrame,
                         reference_date: str = "2022-08-30") -> DataFrame:
    """
    Aggregate loan offer data per customer ID.

    Computes the total number of offers, average inter-offer time, declined
    and success outcome counts, average amount offered, days since the last
    offer, and a binary LABEL (1 if the customer ever took up an offer).

    Args:
        loans_df:       Raw loans DataFrame (ID, DATE, OUTCOME, AMOUNT).
        reference_date: Date string used to compute DAYS_SINCE_LAST_OFFER.
                        Defaults to the last date in the transaction data.

    Returns:
        DataFrame with one row per customer ID containing loan-derived
        feature columns.
    """
    w_outcome = Window.partitionBy("ID")
    w_date = Window.partitionBy("ID").orderBy("DATE")

    loans_df = (
        loans_df
        .withColumn(
            "DECLINED_OUTCOME",
            F.count(F.when(F.col("OUTCOME") == "Declined", 1)).over(w_outcome),
        )
        .withColumn(
            "SUCCESS_OUTCOME",
            F.count(F.when(F.col("OUTCOME") == "TakeUp", 1)).over(w_outcome),
        )
        .withColumn(
            "INTER_OFFER_TIME_DAYS",
            F.datediff(F.col("DATE"), F.lag("DATE").over(w_date)),
        )
    )

    loans_df = (
        loans_df
        .groupBy("ID")
        .agg(
            F.count("*").alias("TOTAL_OFFERS"),
            F.avg("INTER_OFFER_TIME_DAYS").alias("AVG_INTER_OFFER_TIME_DAYS"),
            F.mode("DECLINED_OUTCOME").alias("DECLINED_OUTCOMES"),
            F.mode("SUCCESS_OUTCOME").alias("SUCCESS_OUTCOME"),
            F.avg("AMOUNT").alias("AVG_AMOUNT_OFFERED"),
            F.max("DATE").alias("LAST_OFFER_DATE"),
        )
    )

    loans_df = (
        loans_df
        .withColumn(
            "DAYS_SINCE_LAST_OFFER",
            F.datediff(F.lit(reference_date), F.col("LAST_OFFER_DATE")),
        )
        .drop("LAST_OFFER_DATE")
        .withColumn(
            "LABEL",
            F.when(F.col("SUCCESS_OUTCOME") > 0, 1).otherwise(0),
        )
    )

    return loans_df


# --------------------------------------------------------------------------- #
# Transaction feature nodes                                                    #
# --------------------------------------------------------------------------- #

def create_transaction_features(
    transactions_df: DataFrame,
    customers_df: DataFrame,
    analysis_level: str = "MONTH",
) -> DataFrame:
    """
    Aggregate transaction data per customer ID at the specified time level.

    Joins transactions with customer attributes, then computes per-period
    metrics (end balance, transaction counts, income/expense totals, overdraft
    usage, additional income, living means, income-spent ratio) and
    aggregates them to a single summary row per customer.

    Args:
        transactions_df: Prepared transactions DataFrame (from data_processing pipeline).
        customers_df:     Prepared customers DataFrame (from data_processing pipeline).
        analysis_level:   Time bucket for the first aggregation pass — one of
                          "MONTH", "WEEK", or "DAY".  Defaults to "MONTH".

    Returns:
        DataFrame with one row per customer ID containing transaction-derived
        feature columns plus GENDER, AGE, and INCOME.
    """
    transactions_df = (
        transactions_df
        .join(customers_df, on="ID", how="full")
        .withColumn("MONTH", F.date_format("DATE", "MM"))
        .withColumn("WEEK", F.weekofyear("DATE"))
        .withColumn("DAY", F.date_format("DATE", "dd"))
    )

    # Per-period aggregation -------------------------------------------------- #
    analysis_level_transactions = (
        transactions_df
        .groupBy("ID", analysis_level, "GENDER", "AGE", "INCOME")
        .agg(
            F.max_by("BALANCE", "__index_level_0__").alias("END_BALANCE"),
            F.sum(F.when(F.col("AMOUNT") < 0, 1).otherwise(0)).alias("TRANS_COUNT"),
            F.sum(F.when(F.col("AMOUNT") > 0, 1).otherwise(0)).alias("INCOME_COUNT"),
            F.avg(F.when(F.col("AMOUNT") < 0, F.col("AMOUNT")).otherwise(F.lit(None))).alias("AVG_TRANS_AMOUNT"),
            F.sum(F.when(F.col("AMOUNT") > 0, F.col("AMOUNT")).otherwise(F.lit(None))).alias("TOTAL_INCOME"),
            F.sum(F.when(F.col("AMOUNT") < 0, F.col("AMOUNT")).otherwise(F.lit(None))).alias("TOTAL_EXPENSES"),
            F.avg(
                F.when(F.col("AMOUNT") == F.col("INCOME"), F.col("DAY").cast("int"))
                .otherwise(F.lit(None))
            ).alias("INCOME_DAY"),
            F.sum(
                F.when((F.col("AMOUNT") < 0) & (F.col("BALANCE") < 0), 1).otherwise(0)
            ).alias("OVERDRAFT_SPENT"),
            F.sum(
                F.when((F.col("AMOUNT") < 0) & (F.col("BALANCE") < 0), F.col("AMOUNT"))
                .otherwise(F.lit(None))
            ).alias("OVERDRAFT_AMOUNT"),
        )
    )

    # Derived columns --------------------------------------------------------- #
    analysis_level_transactions = (
        analysis_level_transactions
        .withColumn(
            "ADD_INCOME",
            F.when(F.col("TOTAL_INCOME") == 0, F.lit(None))
            .otherwise(
                F.when(F.col("TOTAL_INCOME") > F.col("INCOME"), F.col("TOTAL_INCOME") - F.col("INCOME"))
                .otherwise(0)
            ),
        )
        .withColumn(
            "LIVING_MEANS",
            F.when(F.col("TOTAL_INCOME") == 0, F.lit(None))
            .otherwise(F.col("TOTAL_INCOME") - F.col("TOTAL_EXPENSES")),
        )
        .withColumn(
            "INCOME_SPENT_%",
            F.when(F.col("TOTAL_INCOME") == 0, F.lit(None))
            .otherwise((-F.col("TOTAL_EXPENSES") / F.col("INCOME")) * 100),
        )
    )

    # Final per-customer aggregation ------------------------------------------ #
    transaction_summary = (
        analysis_level_transactions
        .groupBy("ID", "GENDER", "AGE", "INCOME")
        .agg(
            F.sum(F.when(F.col("END_BALANCE") > 0, 1).otherwise(0)).alias("POSITIVE_END_BALANCE"),
            F.sum(F.when(F.col("END_BALANCE") < 0, 1).otherwise(0)).alias("NEGATIVE_END_BALANCE"),
            F.avg("END_BALANCE").alias("AVG_MONTHLY_END_BALANCE"),
            F.stddev("END_BALANCE").alias("STD_MONTHLY_END_BALANCE"),
            F.avg("TRANS_COUNT").alias("AVG_TRANS_COUNT"),
            F.stddev("TRANS_COUNT").alias("STD_TRANS_COUNT"),
            F.avg("INCOME_COUNT").alias("AVG_INCOME_COUNT"),
            F.stddev("INCOME_COUNT").alias("STD_INCOME_COUNT"),
            F.avg("AVG_TRANS_AMOUNT").alias("AVG_TRANS_AMOUNT"),
            F.stddev("AVG_TRANS_AMOUNT").alias("STD_TRANS_AMOUNT"),
            F.avg("TOTAL_INCOME").alias("AVG_TOTAL_INCOME"),
            F.stddev("TOTAL_INCOME").alias("STD_TOTAL_INCOME"),
            F.avg("TOTAL_EXPENSES").alias("AVG_TOTAL_EXPENSES"),
            F.stddev("TOTAL_EXPENSES").alias("STD_TOTAL_EXPENSES"),
            F.mode("INCOME_DAY").alias("INCOME_DAY"),
            F.avg("OVERDRAFT_AMOUNT").alias("AVG_OVERDRAFT_SPENT"),
            F.sum("OVERDRAFT_SPENT").alias("OVERDRAFT_SPENT"),
            F.avg("ADD_INCOME").alias("AVG_ADD_INCOME"),
            F.sum("ADD_INCOME").alias("ADD_INCOME"),
            F.avg("LIVING_MEANS").alias("LIVING_MEANS"),
            F.avg("INCOME_SPENT_%").alias("INCOME_SPENT_%"),
        )
        .fillna(0)
    )

    return transaction_summary


# --------------------------------------------------------------------------- #
# Consolidation node                                                           #
# --------------------------------------------------------------------------- #

def consolidate(transaction_summary: DataFrame,
                 loan_features: DataFrame) -> DataFrame:
    """
    Outer-join the transaction summary with loan features on customer ID.

    Null values introduced by the outer join are filled with 0.

    Args:
        transaction_summary: Output of create_transaction_features().
        loan_features:        Output of create_loan_features().

    Returns:
        DataFrame with one row per customer ID containing all consolidated
        feature columns.
    """
    return (
        transaction_summary
        .join(loan_features, on="ID", how="outer")
        .fillna(0)
    )


# --------------------------------------------------------------------------- #
# Feature engineering nodes                                                    #
# --------------------------------------------------------------------------- #

def encode_categorical(consolidated_df: DataFrame) -> DataFrame:
    """
    Encode the GENDER column from string values to integers.

    Maps 'M' → 0 and 'F' → 1, matching the encoding used in the exploratory
    notebook.

    Args:
        consolidated_df: Consolidated DataFrame from the consolidate() node.

    Returns:
        DataFrame with GENDER encoded as an integer column.
    """
    return consolidated_df.withColumn(
        "GENDER",
        F.when(F.col("GENDER") == "M", 0).otherwise(1),
    )


def remove_correlated_features(df: DataFrame,
                                correlation_threshold: float = 0.80) -> DataFrame:
    """
    Remove highly correlated (redundant) numeric features.

    Computes the absolute correlation matrix of all numeric columns (excluding
    ID and LABEL), identifies pairs whose correlation exceeds the threshold,
    and drops the first column of each redundant pair.

    The data (≈2 500 rows) is small enough to convert to pandas for the
    correlation computation, then converted back to Spark.

    Args:
        df:                     Encoded DataFrame.
        correlation_threshold:  Pairs with |corr| above this value are considered
                               redundant.  Defaults to 0.80.

    Returns:
        Spark DataFrame with redundant columns removed.
    """
    spark = df.sparkSession
    pdf = df.toPandas()

    cols_to_exclude = ["ID", "LABEL"]
    feature_cols = [c for c in pdf.columns if c not in cols_to_exclude]

    corr_matrix = pdf[feature_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    redundant_features = [
        column
        for column in upper.columns
        for row in upper.index
        if upper.loc[row, column] > correlation_threshold
    ]

    print(f"Removing {len(redundant_features)} redundant features: "
          f"{redundant_features}")

    pdf_reduced = pdf.drop(columns=redundant_features)
    return spark.createDataFrame(pdf_reduced)


def transform_skewed_features(df: DataFrame) -> DataFrame:
    """
    Apply skewness-based transformations to numeric feature columns.

    For each numeric column (excluding ID, LABEL, AGE, INCOME_SPENT_%, and
    GENDER), the skewness is computed.  Columns are then transformed in place:

    | Skewness      | Direction | Transform              |
    |---------------|-----------|------------------------|
    | > 1           | positive  | log1p                  |
    | 0.5 – 1       | positive  | sqrt                   |
    | 0 – 0.5       | positive  | none                   |
    | < -1          | negative  | reflect + log1p        |
    | -1 – 0        | negative  | square                 |

    Args:
        df: DataFrame with correlated features already removed.

    Returns:
        Spark DataFrame with skewed columns transformed.
    """
    spark = df.sparkSession
    pdf = df.toPandas()

    numeric_cols = pdf.select_dtypes(include=["float64", "int64"]).columns
    cols_to_exclude = ["ID", "LABEL", "AGE", "INCOME_SPENT_%", "GENDER"]
    skew_cols = [c for c in numeric_cols if c not in cols_to_exclude]

    for col in skew_cols:
        s = skew(pdf[col].dropna())

        if s > 1:
            pdf[col] = np.log1p(pdf[col].clip(lower=0))
            print(f"{col}: strong positive skew ({s:.2f}) -> log1p")
        elif 0.5 < s <= 1:
            pdf[col] = np.sqrt(pdf[col].clip(lower=0))
            print(f"{col}: moderate positive skew ({s:.2f}) -> sqrt")
        elif 0 < s <= 0.5:
            print(f"{col}: light positive skew ({s:.2f}) -> no transform")
        elif s < -1:
            max_val = pdf[col].max()
            pdf[col] = np.log1p(max_val - pdf[col])
            print(f"{col}: strong negative skew ({s:.2f}) -> reflect + log1p")
        elif -1 <= s < 0:
            pdf[col] = pdf[col] ** 2
            print(f"{col}: moderate negative skew ({s:.2f}) -> square")

    return spark.createDataFrame(pdf)
