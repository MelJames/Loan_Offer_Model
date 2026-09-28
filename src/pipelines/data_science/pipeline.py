"""
Data science pipeline for the Loan Offer Model.

This module wires the feature-engineering nodes together in a Kedro-style
sequence, using the project's own Catalog class (src.catalog_loader.Catalog)
for loading prepared data and saving consolidated / feature results.

Usage (from a notebook or script):
    from src.pipelines.data_science.pipeline import run_pipeline
    run_pipeline(spark, catalog_path)

Or step-by-step:
    from src.pipelines.data_science.pipeline import DataSciencePipeline
    pipeline = DataSciencePipeline(spark, catalog_path)
    pipeline.run()
"""

from pyspark.sql import SparkSession

from src.catalog_loader import Catalog
from src.pipelines.data_science import nodes


class DataSciencePipeline:
    """
    Orchestrates the data science (feature engineering) pipeline.

    Nodes are registered as (name, callable, inputs, outputs) tuples and
    executed in order.  Each node receives DataFrames loaded from the catalog
    (or outputs of previous nodes) and writes its result back to the catalog.
    """

    def __init__(self, spark: SparkSession, catalog_path: str):
        self.spark = spark
        self.catalog = Catalog(spark, catalog_path)

        # ------------------------------------------------------------------ #
        # Node registry – each entry is (name, func, inputs, outputs)        #
        #   name:    human-readable label for logging                        #
        #   func:    the node function from nodes.py                          #
        #   inputs:  list of catalog keys or node-output aliases to load     #
        #   outputs: list of catalog keys to save the result to               #
        # ------------------------------------------------------------------ #
        self.nodes = [
            {
                "name": "create_loan_features",
                "func": nodes.create_loan_features,
                "inputs": ["loans"],
                "outputs": ["_loan_features"],
            },
            {
                "name": "create_transaction_features",
                "func": nodes.create_transaction_features,
                "inputs": ["prep_transactions", "prep_customer"],
                "outputs": ["_transaction_summary"],
            },
            {
                "name": "consolidate",
                "func": nodes.consolidate,
                "inputs": ["_transaction_summary", "_loan_features"],
                "outputs": ["consolidated"],
            },
            {
                "name": "encode_categorical",
                "func": nodes.encode_categorical,
                "inputs": ["consolidated"],
                "outputs": ["_encoded"],
            },
            {
                "name": "remove_correlated_features",
                "func": nodes.remove_correlated_features,
                "inputs": ["_encoded"],
                "outputs": ["_reduced"],
            },
            {
                "name": "transform_skewed_features",
                "func": nodes.transform_skewed_features,
                "inputs": ["_reduced"],
                "outputs": ["features"],
            },
        ]

        # In-memory store for intermediate results that are not persisted
        self._intermediate = {}

    def _resolve_input(self, key: str):
        """
        Resolve an input key to a DataFrame.

        If the key starts with '_', it is an intermediate result produced by
        a previous node and stored in self._intermediate.  Otherwise, load it
        from the catalog.
        """
        if key.startswith("_"):
            return self._intermediate[key]
        return self.catalog.load(key)

    def _resolve_output(self, key: str, df):
        """
        Store or save a node's output.

        Intermediate keys (prefixed with '_') are kept in memory for downstream
        nodes.  Catalog keys are persisted via catalog.save().
        """
        if key.startswith("_"):
            self._intermediate[key] = df
        else:
            self.catalog.save(key, df)

    def run(self):
        """Execute all nodes in order."""
        for node in self.nodes:
            name = node["name"]
            func = node["func"]
            inputs = node["inputs"]
            outputs = node["outputs"]

            print(f"--- Running node: {name} ---")
            print(f"    inputs:  {inputs}")
            print(f"    outputs: {outputs}")

            # Load inputs
            args = [self._resolve_input(k) for k in inputs]

            # Execute node
            result = func(*args)

            # Store outputs
            for out_key in outputs:
                self._resolve_output(out_key, result)
                if not out_key.startswith("_"):
                    row_count = result.count()
                    print(f"    Saved '{out_key}' to catalog ({row_count} rows)")

            print(f"    Node '{name}' completed.\n")

        print("=== Data science pipeline finished ===")


def run_pipeline(spark: SparkSession, catalog_path: str):
    """
    Convenience function to run the full data science pipeline.

    Args:
        spark:        Active SparkSession.
        catalog_path: Path to catalog.yml.
    """
    pipeline = DataSciencePipeline(spark, catalog_path)
    pipeline.run()
    return pipeline
