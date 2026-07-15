"""Tests for the growing benchmark table.

The property that matters is idempotency: re-running a notebook must REPLACE its row, not
append a near-duplicate, or the "single growing table" silently becomes a pile of runs.
"""
import numpy as np
import pandas as pd
import pytest

from src.eval.benchmark import BenchmarkRow, BenchmarkTable, build_row
from src.eval.eval_config import BenchmarkConfig
from src.eval.metrics import MCPricingError, SurfaceError


@pytest.fixture
def table(tmp_path) -> BenchmarkTable:
    return BenchmarkTable(BenchmarkConfig(csv_path=str(tmp_path / "benchmark.csv"),
                                          md_path=str(tmp_path / "benchmark.md")))


def _row(model: str = "heston-nuts", snapshot: str = "2026-07-11", **kwargs) -> BenchmarkRow:
    defaults = {
        "n_quotes": 690, "iv_rmse_train": 0.0106, "iv_rmse_test": 0.0152,
        "price_rmse": 15.43, "svi_iv_rmse": 0.0031, "w1_rel_mean": 0.0042,
        "mc_price_rmse": 0.21, "mc_price_z_max": 1.8, "martingale_rel_max": 0.0009,
        "feller": False, "r_hat_max": 1.004, "ess_min": 812.0, "divergences": 0,
    }

    return BenchmarkRow(model=model, snapshot=snapshot, **(defaults | kwargs))


def test_load_on_a_missing_file_gives_an_empty_frame_with_the_right_columns(
    table: BenchmarkTable,
) -> None:
    empty = table.load()
    assert empty.empty
    assert list(empty.columns) == BenchmarkTable.columns()


def test_upsert_writes_both_the_csv_store_and_the_markdown(table: BenchmarkTable) -> None:
    table.upsert(_row())

    assert table.csv_path.exists()
    assert table.md_path.exists()
    assert "heston-nuts" in table.md_path.read_text()


def test_rerunning_the_same_model_on_the_same_snapshot_replaces_its_row(
    table: BenchmarkTable,
) -> None:
    """The idempotency contract: one model on one chain is ONE result, however many times the
    notebook is executed."""
    table.upsert(_row(iv_rmse_test=0.0152))
    out = table.upsert(_row(iv_rmse_test=0.0139))

    assert len(out) == 1
    assert out.loc[0, "iv_rmse_test"] == 0.0139        # the newer value won


def test_a_different_snapshot_or_model_adds_a_row(table: BenchmarkTable) -> None:
    table.upsert(_row(model="heston-nuts", snapshot="2026-07-11"))
    table.upsert(_row(model="heston-nuts", snapshot="2026-08-01"))
    out = table.upsert(_row(model="heston-mse", snapshot="2026-07-11"))

    assert len(out) == 3
    assert list(out["model"]) == ["heston-mse", "heston-nuts", "heston-nuts"]   # sorted by key


def test_values_survive_the_csv_round_trip(table: BenchmarkTable) -> None:
    table.upsert(_row(iv_rmse_test=0.0152341234))
    reloaded = table.load()

    assert reloaded.loc[0, "iv_rmse_test"] == pytest.approx(0.0152341234, abs=1e-12)
    assert bool(reloaded.loc[0, "feller"]) is False


def test_render_emits_a_header_a_separator_and_one_line_per_row() -> None:
    frame = pd.DataFrame([vars(_row()), vars(_row(snapshot="2026-08-01"))])
    lines = [ln for ln in BenchmarkTable.render(frame).splitlines() if ln.startswith("|")]

    assert len(lines) == 2 + 2                          # header + separator + 2 rows
    assert lines[0].startswith("| model | snapshot |")


def test_render_shows_missing_mcmc_fields_as_a_dash() -> None:
    """A least-squares model has no R-hat; the cell must read as absent, not as 0.0."""
    frame = pd.DataFrame([vars(_row(r_hat_max=None, ess_min=None, divergences=None))])
    rendered = BenchmarkTable.render(frame)

    assert "—" in rendered


def test_build_row_assembles_from_the_metric_objects() -> None:
    train = SurfaceError(n_quotes=500, n_dropped=0, iv_rmse=0.010, iv_mae=0.008,
                         price_rmse=14.0, price_rel_mae=0.04)
    test = SurfaceError(n_quotes=190, n_dropped=2, iv_rmse=0.015, iv_mae=0.012,
                        price_rmse=19.0, price_rel_mae=0.06)
    mc = MCPricingError(n_quotes=690, rmse=0.21, rel_rmse=0.002, max_abs_error=0.5,
                        mean_stderr=0.15, z_max=1.8)
    marginals = pd.DataFrame({"w1_rel": [0.003, 0.005]})
    martingale = pd.DataFrame({"rel_residual": [0.0004, -0.0009]})
    posterior = pd.DataFrame({"r_hat": [1.001, 1.004], "n_eff": [900.0, 812.0]})
    posterior.attrs["divergences"] = 3

    row = build_row("heston-nuts", "2026-07-11", train=train, test=test,
                    svi_iv_rmse=0.0031, mc=mc, marginals=marginals,
                    martingale=martingale, feller=False, posterior=posterior)

    assert row.n_quotes == 690                          # train + test
    assert row.iv_rmse_test == 0.015                    # the headline is the HELD-OUT number
    assert row.w1_rel_mean == pytest.approx(0.004)
    assert row.martingale_rel_max == pytest.approx(0.0009)   # max of the ABSOLUTE residual
    assert row.r_hat_max == 1.004
    assert row.ess_min == 812.0
    assert row.divergences == 3


def test_build_row_without_a_posterior_leaves_the_mcmc_fields_empty() -> None:
    err = SurfaceError(n_quotes=10, n_dropped=0, iv_rmse=0.01, iv_mae=0.01,
                       price_rmse=1.0, price_rel_mae=0.01)
    mc = MCPricingError(n_quotes=10, rmse=0.1, rel_rmse=0.001, max_abs_error=0.2,
                        mean_stderr=0.1, z_max=2.0)
    row = build_row("heston-mse", "2026-07-11", train=err, test=err, svi_iv_rmse=0.003,
                    mc=mc, marginals=pd.DataFrame({"w1_rel": [0.004]}),
                    martingale=pd.DataFrame({"rel_residual": [np.float64(0.001)]}),
                    feller=True)

    assert row.r_hat_max is None
    assert row.ess_min is None
    assert row.divergences is None
