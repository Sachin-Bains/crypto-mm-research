"""Walk-forward evaluation of short-horizon forecasts.

The data is split into consecutive blocks (1 hour by default). Each block is
predicted by models trained only on earlier blocks, with a gap of the longest
horizon between the end of training and the start of testing so that no
training target overlaps the test period. Nothing is shuffled.

Metrics, all out of sample:
  r2     1 - sum((y - pred)^2) / sum(y^2): improvement over predicting no change.
         Zero is the honest baseline here, not the sample mean, because a
         live system cannot know the mean of the future.
  ic     correlation between prediction and outcome
  hit    share of the 10% most confident predictions that got the direction right

Rows are 100 ms apart but targets span up to 30 s, so neighbouring rows are
far from independent. Treat these numbers as estimates over hours of data,
not over hundreds of thousands of independent samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .features import FEATURES, HORIZONS_S, TARGETS

GAP_NS = max(HORIZONS_S) * 1_000_000_000
MIN_TRAIN_BLOCKS = 2


def load(paths: list[Path]) -> dict[str, np.ndarray]:
    tbl = pa.concat_tables([pq.read_table(p) for p in paths])
    return {c: tbl.column(c).to_numpy() for c in tbl.column_names}


class Ridge:
    """Linear regression on standardised features with an L2 penalty (closed form)."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.w = np.linalg.solve(A, Z.T @ y)
        self.b = 0.0  # no intercept: the baseline is "no change", and drift is not forecastable
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.w + self.b


class Boosted:
    """Gradient-boosted trees (LightGBM), deliberately small and heavily regularised."""

    def __init__(self, rounds: int = 300) -> None:
        import lightgbm as lgb  # optional dependency
        self.lgb, self.rounds = lgb, rounds

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Boosted":
        params = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=2000,
                      feature_fraction=0.8, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0, verbose=-1)
        self.m = self.lgb.train(params, self.lgb.Dataset(X, y), num_boost_round=self.rounds)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.m.predict(X)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if len(y) < 100:
        return {"n": int(len(y))}
    r2 = 1 - np.sum((y - p) ** 2) / np.sum(y ** 2)
    ic = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else 0.0
    top = np.abs(p) >= np.quantile(np.abs(p), 0.9)
    moved = top & (y != 0)
    hit = float(np.mean(np.sign(p[moved]) == np.sign(y[moved]))) if moved.any() else float("nan")
    return {"n": int(len(y)), "r2": round(float(r2), 5), "ic": round(ic, 4), "hit_top10": round(hit, 4)}


def walk_forward(d: dict, block_ns: int, models: dict, subsample: int = 5):
    t = d["recv_ns"]
    X_all = np.column_stack([d[f] for f in FEATURES]).astype(np.float64)
    feat_ok = np.isfinite(X_all).all(1)
    blocks = (t - t[0]) // block_ns
    preds = {(m, tg): np.full(len(t), np.nan) for m in models for tg in TARGETS}
    for b in np.unique(blocks):
        if b < MIN_TRAIN_BLOCKS:
            continue
        test = blocks == b
        t_start = t[test][0]
        train = t < t_start - GAP_NS
        for tg in TARGETS:
            y = d[tg].astype(np.float64)
            tr = train & feat_ok & np.isfinite(y)
            te = test & feat_ok & np.isfinite(y)
            if tr.sum() < 10_000 or te.sum() < 100:
                continue
            tr_idx = np.flatnonzero(tr)[::subsample]   # neighbouring rows are near-duplicates
            for name, make in models.items():
                m = make().fit(X_all[tr_idx], y[tr_idx])
                preds[(name, tg)][te] = m.predict(X_all[te])
    return preds


def univariate_ic(d: dict) -> dict:
    """Correlation of each feature with each target over all rows (descriptive, in sample)."""
    out = {}
    for f in FEATURES:
        x = d[f].astype(np.float64)
        out[f] = {}
        for tg in TARGETS:
            y = d[tg].astype(np.float64)
            ok = np.isfinite(x) & np.isfinite(y)
            out[f][tg] = round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 4) if ok.sum() > 100 else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True, help="comma-separated UTC dates with feature files")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--block-hours", type=float, default=1.0)
    ap.add_argument("--no-boost", action="store_true")
    args = ap.parse_args()

    days = [x.strip() for x in args.dates.split(",") if x.strip()]
    d = load([Path(args.features) / f"binance_BTCUSDT_{day}.parquet" for day in days])
    print(f"{len(d['recv_ns']):,} rows ({len(d['recv_ns']) / 36000:.1f} h) from {', '.join(days)}")

    models = {"ridge": Ridge}
    if not args.no_boost:
        try:
            import lightgbm  # noqa: F401
            models["boosted"] = Boosted
        except ImportError:
            print("lightgbm not installed: running ridge only")

    preds = walk_forward(d, int(args.block_hours * 3.6e12), models)

    report = {"rows": int(len(d["recv_ns"])), "dates": days, "oos": {}, "univariate_ic": univariate_ic(d)}
    print("\nOut-of-sample, walk-forward (train on earlier hours only)")
    print(f"{'model':<9}{'target':<7}{'rows':>10}{'r2':>10}{'ic':>9}{'hit (top 10%)':>16}")
    for (name, tg), p in preds.items():
        y = d[tg].astype(np.float64)
        ok = np.isfinite(p) & np.isfinite(y)
        m = metrics(y[ok], p[ok])
        report["oos"][f"{name}/{tg}"] = m
        if "r2" in m:
            print(f"{name:<9}{tg:<7}{m['n']:>10,}{m['r2']:>10.4f}{m['ic']:>9.3f}{m['hit_top10']:>16.1%}")
        else:
            print(f"{name:<9}{tg:<7}  not enough data (need {MIN_TRAIN_BLOCKS + 1}+ blocks)")

    print("\nFeature correlation with each target (all rows, descriptive)")
    print(f"{'feature':<14}" + "".join(f"{tg:>9}" for tg in TARGETS))
    for f, row in report["univariate_ic"].items():
        print(f"{f:<14}" + "".join(f"{(v if v is not None else float('nan')):>9.3f}" for v in row.values()))

    out = Path("data/results")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{days[0]}_{days[-1]}"
    (out / f"forecast_{tag}.json").write_text(json.dumps(report, indent=2))
    cols = {"recv_ns": d["recv_ns"], "mid": d["mid"], "rv_30s": d["rv_30s"], "spread_bps": d["spread_bps"]}
    cols.update({tg: d[tg] for tg in TARGETS})
    cols.update({f"pred_{n}_{tg}": p.astype(np.float32) for (n, tg), p in preds.items()})
    pq.write_table(pa.table(cols), out / f"forecast_oos_{tag}.parquet", compression="zstd")
    print(f"\nSaved report and out-of-sample predictions to {out}")


if __name__ == "__main__":
    main()
