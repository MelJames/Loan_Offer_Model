"""
Data processing nodes for the Loan Offer Model pipeline.

Each function takes one or more Spark DataFrames (and optional parameters)
and returns a transformed DataFrame.  The pipeline.py module wires these
nodes together in sequence using the project's Catalog class for I/O.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# --------------------------------------------------------------------------- #
# Customer cleaning nodes                                                     #
# --------------------------------------------------------------------------- #

def impute_missing_income(customers_df: DataFrame, transactions_df: DataFrame) -> DataFrame:
    """
    Impute missing INCOME values using the mode of positive transaction
    amounts per customer.

    For every customer whose INCOME is null, join to their positive-amount
    transactions and take the mode of the AMOUNT column as the imputed monthly
    income.  Customers with no positive transactions remain null.

    Args:
        customers_df:     Raw customers DataFrame (ID, GENDER, AGE, INCOME).
        transactions_df:  Raw transactions DataFrame (ID, DATE, AMOUNT, BALANCE, ...).

    Returns:
        DataFrame with the same schema as customers_df, with null INCOME
        values replaced by the mode of positive transaction amounts.
    """
    imputed_income = (
        customers_df
        .filter(F.col("INCOME").isNull())
        .select("ID")
        .join(
            transactions_df.filter(F.col("AMOUNT") > 0),
            on="ID",
            how="left",
        )
        .groupBy("ID")
        .agg(F.mode(F.col("AMOUNT")).alias("AMOUNT"))
    )

    return (
        customers_df
        .join(imputed_income, on="ID", how="left")
        .withColumn(
            "INCOME",
            F.when(F.col("INCOME").isNull(), F.col("AMOUNT")).otherwise(F.col("INCOME")),
        )
        .drop("AMOUNT")
    )


def impute_unrealistic_age(customers_df: DataFrame,
                           age_threshold: int = 100) -> DataFrame:
    """
    Replace unrealistic ages (>= age_threshold) with the median age of
    customers in the same income bracket and gender.

    The income bracket is determined by binning the INCOME column into 30
    equal-width buckets.  For each customer with an unrealistic age, the
    median AGE of all customers sharing the same bracket and GENDER is
    computed and used as the imputed value.

    Args:
        customers_df:  Customers DataFrame (already has income imputed).
        age_threshold: Ages at or above this value are treated as unrealistic.
                       Defaults to 100.

    Returns:
        DataFrame with the same schema, unrealistic AGE values replaced by
        the bracket median.
    """
    # Assign each customer to an income bracket (0-29)
    df = customers_df.withColumn(
        "income_bracket",
        F.when(F.col("INCOME").isNull(), F.lit(None).cast("int"))
        .otherwise((F.col("INCOME") / (F.max("INCOME").over(Window.partitionBy("GENDER")) / F.lit(30))).cast("int")),
    )

    # Compute median age per (bracket, gender) group
    bracket_median = (
        df.filter(F.col("AGE") < age_threshold)
        .groupBy("income_bracket", "GENDER")
        .agg(F.median(F.col("AGE")).alias("median_age"))
    )

    # Join back and impute
    return (
        df
        .join(bracket_median, on=["income_bracket", "GENDER"], how="left")
        .withColumn(
            "AGE",
            F.when(
                (F.col("AGE") >= age_threshold) & F.col("median_age").isNotNull(),
                F.col("median_age"),
            ).otherwise(F.col("AGE")),
        )
        .drop("income_bracket", "median_age")
    )


# --------------------------------------------------------------------------- #
# Transaction cleaning nodes                                                  #
# --------------------------------------------------------------------------- #

def impute_null_balances(transactions_df: DataFrame,
                         order_col: str = "__index_level_0__") -> DataFrame:
    """
    Forward-fill null BALANCE values using the last known balance adjusted
    by cumulative transaction amounts.

    For consecutive rows with a null BALANCE, the imputed value is:
        last_non_null_balance + cumulative_sum(AMOUNT) - first_amount_in_group

    where the "group" starts at the row holding the last non-null BALANCE
    and extends through the following null rows until the next non-null
    BALANCE is encountered.

    Args:
        transactions_df: Raw transactions DataFrame.
        order_col:       Column used to order transactions within each ID.
                         Defaults to "__index_level_0__" (the pandas index
                         carried over from the original parquet file).

    Returns:
        DataFrame with the same schema, null BALANCE values imputed.
    """
    # Window to carry forward the last non-null balance and create group boundaries
    w_forward = (
        Window
        .partitionBy("ID")
        .orderBy(order_col)
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    df = (
        transactions_df
        .withColumn("last_balance", F.last(F.col("BALANCE"), ignorenulls=True).over(w_forward))
        .withColumn("group_id", F.count(F.when(F.col("BALANCE").isNotNull(), 1)).over(w_forward))
    )

    # Window within each group for cumulative sum of AMOUNT
    w_group = (
        Window
        .partitionBy("ID", "group_id")
        .orderBy(order_col)
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    df = (
        df
        .withColumn("cumsum_amount", F.sum(F.col("AMOUNT")).over(w_group))
        .withColumn("first_amount", F.first(F.col("AMOUNT")).over(w_group))
    )

    # Impute: last_balance + cumulative amounts (excluding the row that had the non-null balance)
    df = (
        df
        .withColumn(
            "BALANCE",
            F.when(
                F.col("BALANCE").isNull(),
                F.col("last_balance") + F.col("cumsum_amount") - F.col("first_amount"),
            ).otherwise(F.col("BALANCE")),
        )
        .drop("last_balance", "group_id", "cumsum_amount", "first_amount")
    )

    return df