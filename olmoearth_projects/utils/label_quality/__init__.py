"""Functions to test label quality."""

from collections.abc import Callable

import geopandas as gpd
from rich.console import Console
from rich.table import Table

from .label_imbalance import label_imbalance
from .spatial_clustering import spatial_clustering
from .spatial_extent import spatial_extent


def check_label_quality(
    df: gpd.GeoDataFrame, checks_to_run: list[str] | None = None
) -> None:
    """Run all label quality checks.

    Args:
        df: the labels to check. Requires a `label` and `geometry` column
        checks_to_run: the checks to run. If not None, must contain at least
            one of {`label_imbalance`, `spatial_clustering`, `spatial_extent`}
    """
    required_columns = ["geometry", "label"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Input df requires `label` and `geometry` column. Missing {col}."
            )

    all_checks: dict[str, Callable] = {
        "label_imbalance": label_imbalance,
        "spatial_clustering": spatial_clustering,
        "spatial_extent": spatial_extent,
    }
    if checks_to_run is not None:
        checks = {k: v for k, v in all_checks.items() if k in checks_to_run}
        if len(checks) == 0:
            raise ValueError(
                "All checks removed. `checks_to_run` must have "
                f"at least one of {all_checks.keys()}"
            )
    else:
        checks = all_checks

    table = Table()

    table.add_column("Check name", justify="right", style="cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", justify="right", style="green")

    table.add_row("", "# instances", str(len(df)))

    for check_name, check_f in checks.items():
        results = check_f(df)
        for result_name, result_value in results.items():
            table.add_row(check_name, result_name, str(result_value))

    console = Console()
    console.print(table)
