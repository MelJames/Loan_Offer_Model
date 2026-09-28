import yaml
from dataclasses import dataclass

@dataclass
class SparkDatasetConfig:
    type: str
    filepath: str
    table: str
    format: str

class Catalog:
    def __init__(self, spark, catalog_path="catalog.yml"):
        with open(catalog_path, "r") as f:
            raw_catalog = yaml.safe_load(f)

        self.spark = spark
        self.catalog = {
            name: SparkDatasetConfig(**cfg)
            for name, cfg in raw_catalog.items()
        }

    def load(self, name):
        cfg = self.catalog[name]

        # Priority: table > filepath
        if cfg.table:
            return self.spark.table(cfg.table)

        return self.spark.read.format(cfg.format).load(cfg.filepath)

    def save(self, name, df):
        cfg = self.catalog[name]
        df.write.format(cfg.format).mode("overwrite").saveAsTable(cfg.table)
