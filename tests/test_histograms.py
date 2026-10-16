
import pandas as pd

from src.eda import plot_numeric_histograms


def test_plot_numeric_histograms_creates_all_plots(tmp_path):
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "salary": [100, 200, 300],
        "city": ["Delhi", "Kolkata", "Delhi"],
    })

    plots = plot_numeric_histograms(
        df,
        output_dir=str(tmp_path),
    )

    assert set(plots.keys()) == {"age", "salary"}
    assert all(path.endswith("_histogram.png") for path in plots.values())
    assert all(tmp_path.joinpath(path.split("/")[-1]).exists() for path in plots.values())
