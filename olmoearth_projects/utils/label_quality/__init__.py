"""Functions to test label quality."""

from collections.abc import Callable

import geopandas as gpd
from rich.console import Console
from rich.table import Table

from .label_imbalance import label_imbalance
from .spatial_clustering import spatial_clustering
from .spatial_extent import spatial_extent


def check_label_quality(df: gpd.GeoDataFrame) -> None:
    """Run all label quality checks."""
    checks: dict[str, Callable] = {
        "label_imbalance": label_imbalance,
        "spatial_clustering": spatial_clustering,
        "spatial_extent": spatial_extent,
    }

    table = Table()

    table.add_column("Check name", justify="right", style="cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", justify="right", style="green")

    for check_name, check_f in checks.items():
        results = check_f(df)
        for result_name, result_value in results.items():
            table.add_row(check_name, result_name, str(result_value))

    console = Console()
    console.print(table)
