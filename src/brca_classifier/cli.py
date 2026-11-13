"""Command-line entry point: ``uv run brca <command> [--config FILE]``.

Commands
--------
eda      class distribution and PCA figures
train    run the experiment grid, save models, metrics and figures
explain  SHAP explanation of the selected (or a given) model
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from . import plots
from .config import DEFAULT_CONFIG, load_config
from .data import load_dataset, select_feature_set, split
from .explain import explain_model
from .training import run_experiment

log = logging.getLogger("brca_classifier")


def _load_split(cfg: dict[str, Any]):
    X, y = load_dataset(cfg["data"])
    return split(X, y, cfg["data"]["test_size"], cfg["seed"])


def _dirs(cfg: dict[str, Any]) -> tuple[Path, Path]:
    results = Path(cfg["output"]["results_dir"])
    models = Path(cfg["output"]["models_dir"])
    results.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return results, models


def cmd_eda(cfg: dict[str, Any]) -> None:
    results_dir, _ = _dirs(cfg)
    X_train, _, y_train, _ = _load_split(cfg)
    fig_dir = results_dir / "figures"
    plots.class_distribution(y_train, fig_dir / "class_distribution.png")
    sets = {name: select_feature_set(X_train, name) for name in cfg["experiments"]["feature_sets"]}
    plots.pca_projection(sets, y_train, fig_dir / "pca_feature_sets.png")
    log.info("EDA figures written to %s", fig_dir)


def _summary_markdown(df: pd.DataFrame, metric: str) -> str:
    cols = {
        "experiment_id": "Experiment",
        f"cv_{metric}_mean": "CV bal. acc.",
        "test_balanced_accuracy": "Test bal. acc. [95% CI]",
        "test_roc_auc": "Test ROC AUC",
        "test_precision": "Test precision",
        "test_recall": "Test recall",
    }
    table = df.sort_values(f"cv_{metric}_mean", ascending=False).copy()
    table[f"cv_{metric}_mean"] = [
        f"{m:.3f} ± {s:.3f}"
        for m, s in zip(table[f"cv_{metric}_mean"], table[f"cv_{metric}_std"], strict=True)
    ]
    table["test_balanced_accuracy"] = [
        f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
        for v, lo, hi in zip(
            table["test_balanced_accuracy"],
            table["test_balanced_accuracy_ci_low"],
            table["test_balanced_accuracy_ci_high"],
            strict=True,
        )
    ]
    for c in ["test_roc_auc", "test_precision", "test_recall"]:
        table[c] = table[c].map("{:.3f}".format)
    table = table[list(cols)].rename(columns=cols)
    header = "| " + " | ".join(table.columns) + " |"
    sep = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in table.itertuples(index=False)]
    return "\n".join([header, sep, *rows]) + "\n"


def _completed(results_dir: Path, models_dir: Path) -> tuple[list[dict], dict]:
    """Rows and params of experiments already saved by a previous (interrupted) run."""
    metrics, params = results_dir / "metrics.csv", results_dir / "best_params.json"
    if not metrics.exists():
        return [], {}
    df = pd.read_csv(metrics)
    df = df[[(models_dir / f"{e}.joblib").exists() for e in df.experiment_id]]
    saved = json.loads(params.read_text()) if params.exists() else {}
    return df.to_dict("records"), {e: saved.get(e, {}) for e in df.experiment_id}


def cmd_train(cfg: dict[str, Any], resume: bool = False) -> None:
    results_dir, models_dir = _dirs(cfg)
    X_train_full, X_test_full, y_train, y_test = _load_split(cfg)
    log.info(
        "Train: %d patients (%d BRCA) | Test: %d patients (%d BRCA) | resampling=%s",
        len(y_train),
        y_train.sum(),
        len(y_test),
        y_test.sum(),
        cfg["resampling"],
    )

    exp = cfg["experiments"]
    rows, params = _completed(results_dir, models_dir) if resume else ([], {})
    for feature_set in exp["feature_sets"]:
        X_train = select_feature_set(X_train_full, feature_set)
        X_test = select_feature_set(X_test_full, feature_set)
        for sampler in exp["samplers"]:
            for model in exp["models"]:
                if f"{feature_set}__{sampler}__{model}" in params:
                    log.info("=== %s | %s | %s === (done, skipped)", feature_set, sampler, model)
                    continue
                log.info("=== %s | %s | %s ===", feature_set, sampler, model)
                for r in run_experiment(
                    feature_set, sampler, model, X_train, y_train, X_test, y_test, cfg
                ):
                    log.info(
                        "  %-8s CV bal.acc %.3f ± %.3f | test bal.acc %.3f",
                        r.model,
                        r.cv_scores["balanced_accuracy_mean"],
                        r.cv_scores["balanced_accuracy_std"],
                        r.test_metrics["balanced_accuracy"],
                    )
                    rows.append(r.to_row())
                    params[r.experiment_id] = r.best_params
                    joblib.dump(r.fitted_model, models_dir / f"{r.experiment_id}.joblib")
                # Write after every experiment so partial results survive interruptions.
                pd.DataFrame(rows).to_csv(results_dir / "metrics.csv", index=False)
                (results_dir / "best_params.json").write_text(
                    json.dumps(params, indent=2, default=str)
                )

    df = pd.DataFrame(rows)
    metric = cfg["selection_metric"]
    best = df.loc[df[f"cv_{metric}_mean"].idxmax()]
    selected = {
        "experiment_id": best["experiment_id"],
        "selection_metric": f"cv_{metric}_mean",
        "cv_score": float(best[f"cv_{metric}_mean"]),
        "test_balanced_accuracy": float(best["test_balanced_accuracy"]),
        "config_resampling": cfg["resampling"],
    }
    (results_dir / "selected_model.json").write_text(json.dumps(selected, indent=2))
    (results_dir / "summary.md").write_text(_summary_markdown(df, metric))
    log.info("Selected by CV: %s", selected)

    # Figures for the selected model.
    fig_dir = results_dir / "figures"
    sel_id = best["experiment_id"]
    same_setup = df[(df.feature_set == best.feature_set) & (df.sampler == best.sampler)]
    models = {e: joblib.load(models_dir / f"{e}.joblib") for e in same_setup.experiment_id}
    X_test_sel = select_feature_set(X_test_full, best["feature_set"])
    plots.confusion(
        models[sel_id],
        X_test_sel,
        y_test,
        f"Confusion matrix\n{sel_id}",
        fig_dir / "confusion_matrix.png",
    )
    plots.roc_curves(
        {r.model: models[r.experiment_id] for r in same_setup.itertuples()},
        {r.model: X_test_sel for r in same_setup.itertuples()},
        y_test,
        fig_dir / "roc_curves.png",
    )
    log.info("Results written to %s", results_dir)


def cmd_explain(cfg: dict[str, Any], experiment_id: str | None) -> None:
    results_dir, models_dir = _dirs(cfg)
    if experiment_id is None:
        selected = json.loads((results_dir / "selected_model.json").read_text())
        experiment_id = selected["experiment_id"]
    feature_set = experiment_id.split("__")[0]
    model = joblib.load(models_dir / f"{experiment_id}.joblib")

    X_train, X_test, _, _ = _load_split(cfg)
    table = explain_model(
        model,
        select_feature_set(X_train, feature_set),
        select_feature_set(X_test, feature_set),
        cfg,
        title=experiment_id,
        out_dir=results_dir / "shap" / experiment_id,
    )
    log.info("Top features:\n%s", table.head(cfg["shap"]["top_n"]).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="brca", description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config file")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("eda", help="class distribution and PCA figures")
    p_train = sub.add_parser("train", help="run the experiment grid")
    p_train.add_argument(
        "--resume", action="store_true", help="skip experiments already saved in results/"
    )
    p_explain = sub.add_parser("explain", help="SHAP explanation of a trained model")
    p_explain.add_argument(
        "--experiment", default=None, help="experiment id (default: model selected by CV)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    cfg = load_config(args.config)

    if args.command == "eda":
        cmd_eda(cfg)
    elif args.command == "train":
        cmd_train(cfg, resume=args.resume)
    elif args.command == "explain":
        cmd_explain(cfg, args.experiment)


if __name__ == "__main__":
    main()
