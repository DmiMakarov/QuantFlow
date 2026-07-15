"""The single growing results table.

PROJECT_PLAN.md §8: "All results land in reports/benchmark.md as a single growing table" and
"the benchmark table itself is a contribution." So it is treated as a build artifact, not as
prose: models write rows into it through `BenchmarkTable.upsert`, and the markdown is rendered
from a csv rather than hand-edited.

Two serialisations of one logical table, on purpose:
  benchmark.csv - the store. Machine-readable, full float precision, what Phase 2/3 will plot.
  benchmark.md  - the deliverable, rendered from the csv.
Parsing markdown back into a DataFrame to make updates idempotent would be a bug farm (float
round-trip drift, NaN rendering, escaping), and pandas' own to_markdown needs `tabulate`,
which is not a project dependency -- so the renderer here is hand-rolled.
"""
from dataclasses import asdict, dataclass, fields
from logging import getLogger
from pathlib import Path

import pandas as pd

from .eval_config import BenchmarkConfig
from .metrics import MCPricingError, SurfaceError

logger = getLogger()

KEY: tuple[str, str] = ("model", "snapshot")

# per-column display precision for the rendered markdown; anything absent falls back to str()
_FORMATS: dict[str, str] = {
    "iv_rmse_train": "{:.4f}", "iv_rmse_test": "{:.4f}", "svi_iv_rmse": "{:.4f}",
    "price_rmse": "{:.3f}", "w1_rel_mean": "{:.5f}", "mc_price_rmse": "{:.4f}",
    "mc_price_z_max": "{:.2f}", "martingale_rel_max": "{:.5f}",
    "r_hat_max": "{:.3f}", "ess_min": "{:.0f}",
}


@dataclass
class BenchmarkRow:
    """One row of reports/benchmark.md: one model, scored on one option-chain snapshot.

    iv_rmse_test is the headline: it is the error on the HELD-OUT wing of strikes, so unlike
    iv_rmse_train it is not something the calibration could optimise directly.

    svi_iv_rmse is the noise floor -- the market interpolant's own in-sample residual. A model
    is not expected to beat it, and one that does is fitting quote noise.

    mc_price_z_max > ~3 means the MC-vs-Fourier gap is a discretisation bias rather than
    sampling noise; below that the mc_price_rmse is just Monte-Carlo error.

    feller is a column, not a rejection: calibrated SPX params routinely violate it, which is
    a fact about the market, not a failed fit.
    """

    model: str
    snapshot: str                    # chain snapshot date, YYYY-MM-DD
    n_quotes: int
    iv_rmse_train: float
    iv_rmse_test: float
    price_rmse: float
    svi_iv_rmse: float
    w1_rel_mean: float
    mc_price_rmse: float
    mc_price_z_max: float
    martingale_rel_max: float
    feller: bool
    r_hat_max: float | None = None   # None for non-MCMC models
    ess_min: float | None = None
    divergences: int | None = None
    run: str = ""                    # ISO timestamp; deliberately NOT part of the key
    notes: str = ""


def build_row(model: str, snapshot: str, *, train: SurfaceError, test: SurfaceError,
              svi_iv_rmse: float, mc: MCPricingError, marginals: pd.DataFrame,
              martingale: pd.DataFrame, feller: bool,
              posterior: pd.DataFrame | None = None, notes: str = "") -> BenchmarkRow:
    """Assemble a BenchmarkRow from the metric objects.

    Pure: takes the metric dataclasses and frames, never a model object. That is what keeps
    this module free of jax/torch and keeps row assembly under test instead of buried in a
    notebook cell.

    train, test - SurfaceError from metrics.surface_rmse on each half of the wing split.
    mc          - MCPricingError from metrics.mc_pricing_error.
    marginals   - metrics.marginal_report output.
    martingale  - metrics.martingale_residual output.
    posterior   - HestonModel.posterior_summary() output, when the model was sampled.
    """
    return BenchmarkRow(
        model=model,
        snapshot=snapshot,
        n_quotes=train.n_quotes + test.n_quotes,
        iv_rmse_train=train.iv_rmse,
        iv_rmse_test=test.iv_rmse,
        price_rmse=train.price_rmse,
        svi_iv_rmse=svi_iv_rmse,
        w1_rel_mean=float(marginals["w1_rel"].mean()),
        mc_price_rmse=mc.rmse,
        mc_price_z_max=mc.z_max,
        martingale_rel_max=float(martingale["rel_residual"].abs().max()),
        feller=feller,
        r_hat_max=None if posterior is None else float(posterior["r_hat"].max()),
        ess_min=None if posterior is None else float(posterior["n_eff"].min()),
        divergences=None if posterior is None else posterior.attrs.get("divergences"),
        notes=notes,
    )


class BenchmarkTable:
    """Insert-or-replace access to the growing results table."""

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        """Resolve where the csv store and the rendered markdown live."""
        cfg = config if config is not None else BenchmarkConfig()
        self.csv_path = Path(cfg.csv_path)
        self.md_path = Path(cfg.md_path)

    @staticmethod
    def columns() -> list[str]:
        """Column order, taken from BenchmarkRow so the two can never drift apart."""
        return [f.name for f in fields(BenchmarkRow)]

    def load(self) -> pd.DataFrame:
        """Read the table so far, or an empty frame with the right columns if it is absent."""
        if not self.csv_path.exists():
            return pd.DataFrame(columns=self.columns())

        return pd.read_csv(self.csv_path)

    def upsert(self, row: BenchmarkRow) -> pd.DataFrame:
        """Insert `row`, or replace the existing row with the same (model, snapshot).

        Keyed rather than appended so that re-executing a notebook does not silently grow a
        pile of near-duplicate rows -- the same model on the same chain is one result, not a
        new one. `run` (the timestamp) sits outside the key precisely so a re-run is
        content-neutral.
        """
        table = self.load()
        incoming = pd.DataFrame([asdict(row)])
        if not table.empty:
            duplicate = ((table["model"] == row.model)
                         & (table["snapshot"].astype(str) == str(row.snapshot)))
            if duplicate.any():
                logger.info("Replacing existing benchmark row for (%s, %s).",
                            row.model, row.snapshot)
            table = table.loc[~duplicate]

        table = pd.concat([table, incoming], ignore_index=True)[self.columns()]
        table = table.sort_values(list(KEY), ignore_index=True)

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.csv_path, index=False)
        self.md_path.write_text(self.render(table))
        logger.info("Benchmark table now holds %d row(s): %s", len(table), self.md_path)

        return table

    @staticmethod
    def _cell(column: str, value: object) -> str:
        """Format one cell, honouring per-column precision and rendering None as an em dash."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        fmt = _FORMATS.get(column)

        return fmt.format(value) if fmt else str(value)

    @classmethod
    def render(cls, table: pd.DataFrame) -> str:
        """Render the table as markdown. Pure, so it is directly testable."""
        columns = [f.name for f in fields(BenchmarkRow)]
        lines = [
            "# QuantFlow benchmark",
            "",
            "Generated by `src.eval.benchmark.BenchmarkTable`. Do not edit by hand — edit the",
            "notebook that produces the row and re-run it (`reports/benchmark.csv` is the store).",
            "",
            "`iv_rmse_test` is the headline: implied-vol RMSE in absolute vol units on the",
            "**held-out wing** of strikes. `svi_iv_rmse` is the market interpolant's own",
            "in-sample residual — the noise floor a model is not expected to beat.",
            "`mc_price_z_max` > 3 means the MC-vs-Fourier gap is discretisation bias, not noise.",
            "",
            "| " + " | ".join(columns) + " |",
            "|" + "|".join(["---"] * len(columns)) + "|",
        ]
        lines.extend(
            "| " + " | ".join(cls._cell(c, r[c]) for c in columns) + " |"
            for _, r in table.iterrows()
        )

        return "\n".join(lines) + "\n"
