"""Create random and scaffold train/validation/test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


DEFAULT_INPUT = Path("data/processed/clean_target_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("data/splits")
DEFAULT_SUMMARY = Path("reports/split_summary.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_ratios(train_size: float, val_size: float, test_size: float) -> None:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}.")


def id_column(df: pd.DataFrame) -> str:
    if "molecule_chembl_id" in df.columns:
        return "molecule_chembl_id"
    if "standard_inchi_key" in df.columns:
        return "standard_inchi_key"
    return "canonical_smiles"


def add_scaffold_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def scaffold_from_smiles(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold or smiles

    df["scaffold"] = df["canonical_smiles"].apply(scaffold_from_smiles)
    return df


def make_random_split(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_end = int(len(shuffled) * train_size)
    val_end = train_end + int(len(shuffled) * val_size)

    return {
        "train": shuffled.iloc[:train_end].copy(),
        "val": shuffled.iloc[train_end:val_end].copy(),
        "test": shuffled.iloc[val_end:].copy(),
    }


def make_scaffold_split(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
) -> dict[str, pd.DataFrame]:
    df = add_scaffold_column(df)
    scaffold_groups = [
        group.sample(frac=1.0, random_state=42)
        for _, group in df.groupby("scaffold", sort=False)
    ]
    scaffold_groups.sort(key=len, reverse=True)

    train_limit = int(len(df) * train_size)
    val_limit = int(len(df) * val_size)

    splits: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    sizes = {"train": 0, "val": 0, "test": 0}

    for group in scaffold_groups:
        if sizes["train"] + len(group) <= train_limit:
            split_name = "train"
        elif sizes["val"] + len(group) <= val_limit:
            split_name = "val"
        else:
            split_name = "test"

        splits[split_name].append(group)
        sizes[split_name] += len(group)

    return {
        name: pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
        for name, parts in splits.items()
    }


def write_splits(splits: dict[str, pd.DataFrame], output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        split_df.to_csv(output_dir / f"{prefix}_{name}.csv", index=False)


def split_stats(splits: dict[str, pd.DataFrame], molecule_id: str) -> pd.DataFrame:
    rows = []
    for name, split_df in splits.items():
        row = {
            "split": name,
            "rows": len(split_df),
            "unique_molecules": split_df[molecule_id].nunique(),
            "pIC50_mean": split_df["pIC50"].mean(),
            "pIC50_std": split_df["pIC50"].std(),
            "pIC50_min": split_df["pIC50"].min(),
            "pIC50_max": split_df["pIC50"].max(),
        }
        if "scaffold" in split_df.columns:
            row["unique_scaffolds"] = split_df["scaffold"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def assert_no_overlap(splits: dict[str, pd.DataFrame], column: str) -> None:
    seen: dict[str, set[str]] = {}
    for name, split_df in splits.items():
        seen[name] = set(split_df[column].dropna().astype(str))

    split_names = list(seen)
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1 :]:
            overlap = seen[left_name] & seen[right_name]
            if overlap:
                raise RuntimeError(
                    f"{column} overlap between {left_name} and {right_name}: "
                    f"{len(overlap)} values."
                )


def build_summary(
    df: pd.DataFrame,
    random_stats: pd.DataFrame,
    scaffold_stats: pd.DataFrame,
    output_dir: Path,
    molecule_id: str,
) -> str:
    return f"""# Krok 3: podziały danych

## Cel

Celem tego kroku było przygotowanie podziałów `train`, `validation` i `test` dla datasetu:

```text
data/processed/clean_target_dataset.csv
```

Przygotowano dwa warianty:

- `random split` - prosty losowy podział, dobry jako pierwszy baseline;
- `scaffold split` - trudniejszy podział chemiczny, gdzie te same scaffoldy nie powinny trafiać do różnych zbiorów.

## Użyte proporcje

```text
train: 80%
validation: 10%
test: 10%
```

Liczba rekordów wejściowych: `{len(df):,}`  
Kolumna identyfikująca cząsteczkę: `{molecule_id}`

## Random split

Pliki:

```text
{output_dir / "random_train.csv"}
{output_dir / "random_val.csv"}
{output_dir / "random_test.csv"}
```

Statystyki:

```text
{random_stats.round(3).to_string(index=False)}
```

## Scaffold split

Pliki:

```text
{output_dir / "scaffold_train.csv"}
{output_dir / "scaffold_val.csv"}
{output_dir / "scaffold_test.csv"}
```

Statystyki:

```text
{scaffold_stats.round(3).to_string(index=False)}
```

## Kontrola przecieków

W obu wariantach sprawdzono, że identyfikatory cząsteczek nie powtarzają się między `train`, `validation` i `test`.

Dla `scaffold split` dodatkowo sprawdzono, że te same wartości `scaffold` nie występują w wielu zbiorach.

## Wniosek

Podziały są gotowe do kolejnego kroku: treningu prostego baseline na deskryptorach RDKit.
"""


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_size, args.val_size, args.test_size)

    df = pd.read_csv(args.input)
    molecule_id = id_column(df)

    random_splits = make_random_split(df, args.train_size, args.val_size, args.seed)
    scaffold_splits = make_scaffold_split(df, args.train_size, args.val_size)

    assert_no_overlap(random_splits, molecule_id)
    assert_no_overlap(scaffold_splits, molecule_id)
    assert_no_overlap(scaffold_splits, "scaffold")

    write_splits(random_splits, args.output_dir, "random")
    write_splits(scaffold_splits, args.output_dir, "scaffold")

    random_stats = split_stats(random_splits, molecule_id)
    scaffold_stats = split_stats(scaffold_splits, molecule_id)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        build_summary(df, random_stats, scaffold_stats, args.output_dir, molecule_id),
        encoding="utf-8",
    )

    print(f"Saved split files in: {args.output_dir}")
    print(f"Saved split summary: {args.summary}")


if __name__ == "__main__":
    main()
