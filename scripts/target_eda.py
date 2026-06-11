"""Run a simple target-specific EDA for the prepared pIC50 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_INPUT = Path("data/processed/clean_target_dataset.csv")
DEFAULT_FIGURES_DIR = Path("reports/figures")
DEFAULT_SUMMARY = Path("reports/target_eda_summary.md")

DESCRIPTOR_COLUMNS = ["MW", "LogP", "TPSA", "HBD", "HBA", "rotatable_bonds", "heavy_atoms"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def save_histogram(df: pd.DataFrame, column: str, path: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(9, 5))
    sns.histplot(df[column].dropna(), bins=40, kde=True, color="#4C78A8")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Liczba cząsteczek")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_descriptor_grid(df: pd.DataFrame, path: Path) -> None:
    columns = [column for column in DESCRIPTOR_COLUMNS if column in df.columns]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.flatten()

    for axis, column in zip(axes, columns):
        sns.histplot(df[column].dropna(), bins=35, color="#72B7B2", ax=axis)
        axis.set_title(column)
        axis.set_xlabel("")
        axis.set_ylabel("")

    for axis in axes[len(columns) :]:
        axis.axis("off")

    fig.suptitle("Rozkłady deskryptorów RDKit", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    columns = ["pIC50"] + [column for column in DESCRIPTOR_COLUMNS if column in df.columns]
    corr = df[columns].corr(method="spearman")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Korelacje Spearmana: pIC50 i deskryptory")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

    return corr


def save_measurement_count_plot(df: pd.DataFrame, path: Path) -> None:
    if "measurement_count" not in df.columns:
        return

    counts = df["measurement_count"].value_counts().sort_index()
    plt.figure(figsize=(9, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#F58518")
    plt.title("Liczba pomiarów na cząsteczkę")
    plt.xlabel("measurement_count")
    plt.ylabel("Liczba cząsteczek")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def dataset_id_column(df: pd.DataFrame) -> str:
    if "molecule_chembl_id" in df.columns:
        return "molecule_chembl_id"
    if "standard_inchi_key" in df.columns:
        return "standard_inchi_key"
    return "canonical_smiles"


def build_summary(df: pd.DataFrame, corr: pd.DataFrame, figures_dir: Path) -> str:
    id_column = dataset_id_column(df)
    target_column = "pref_name" if "pref_name" in df.columns else None
    organism_column = "organism" if "organism" in df.columns else None

    target = df[target_column].iloc[0] if target_column else "unknown"
    organism = df[organism_column].iloc[0] if organism_column else "unknown"

    pIC50 = df["pIC50"].describe().round(3)
    descriptors = df[[column for column in DESCRIPTOR_COLUMNS if column in df.columns]].describe().round(3)
    top_corr = (
        corr["pIC50"]
        .drop("pIC50")
        .abs()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )

    duplicate_smiles = int(df["canonical_smiles"].duplicated().sum()) if "canonical_smiles" in df.columns else 0
    measurement_gt1 = int((df["measurement_count"] > 1).sum()) if "measurement_count" in df.columns else 0

    return f"""# EDA targetu EGFR

## Zakres analizy

Analiza dotyczy przygotowanego datasetu:

```text
data/processed/clean_target_dataset.csv
```

Target:

```text
{target}
{organism}
```

Liczba rekordów: `{len(df):,}`  
Liczba unikalnych identyfikatorów cząsteczek (`{id_column}`): `{df[id_column].nunique():,}`  
Liczba zduplikowanych `canonical_smiles`: `{duplicate_smiles:,}`  
Liczba cząsteczek z więcej niż jednym pomiarem: `{measurement_gt1:,}`

## Rozkład pIC50

Podstawowe statystyki `pIC50`:

```text
{pIC50.to_string()}
```

Interpretacja: wartości `pIC50` są już w skali logarytmicznej, więc nie trzeba ich dodatkowo transformować przed baseline. Większe `pIC50` oznacza silniejszą aktywność wobec targetu.

Wykres:

```text
{figures_dir / "pIC50_distribution.png"}
```

## Deskryptory fizykochemiczne

W zbiorze znajdują się podstawowe deskryptory RDKit: `MW`, `LogP`, `TPSA`, `HBD`, `HBA`, `rotatable_bonds`, `heavy_atoms`.

Statystyki deskryptorów:

```text
{descriptors.to_string()}
```

Wykres:

```text
{figures_dir / "descriptor_distributions.png"}
```

## Korelacje z pIC50

Najsilniejsze bezwzględne korelacje Spearmana z `pIC50` mają:

```text
{", ".join(top_corr)}
```

To nie oznacza jeszcze przyczynowości ani dobrego modelu, ale pomaga zobaczyć, które proste cechy mogą być przydatne dla baseline.

Wykres:

```text
{figures_dir / "descriptor_correlations.png"}
```

## Liczba pomiarów

Kolumna `measurement_count` pokazuje, ile pomiarów źródłowych zostało zagregowanych do jednego rekordu cząsteczki. Większość rekordów ma pojedynczy pomiar, więc `pIC50_std` często wynosi `0`.

Wykres:

```text
{figures_dir / "measurement_count.png"}
```

## Wnioski praktyczne

- Dataset jest wystarczająco mały, żeby trenować prosty baseline bez dużej infrastruktury.
- Dane dotyczą jednego targetu, więc unikamy najważniejszego błędu z poprzedniego EDA: mieszania wielu białek w jednym problemie regresji.
- Najbliższy sensowny krok to przygotowanie splitów `train/validation/test` oraz prostego baseline na deskryptorach RDKit.
"""


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    if "pIC50" not in df.columns:
        raise ValueError("Input dataset must contain a pIC50 column.")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    save_histogram(
        df,
        column="pIC50",
        path=args.figures_dir / "pIC50_distribution.png",
        title="Rozkład pIC50",
        xlabel="pIC50",
    )
    save_descriptor_grid(df, args.figures_dir / "descriptor_distributions.png")
    corr = save_correlation_heatmap(df, args.figures_dir / "descriptor_correlations.png")
    save_measurement_count_plot(df, args.figures_dir / "measurement_count.png")

    args.summary.write_text(build_summary(df, corr, args.figures_dir), encoding="utf-8")

    print(f"Saved EDA summary: {args.summary}")
    print(f"Saved figures in: {args.figures_dir}")


if __name__ == "__main__":
    main()
