"""Prepare a small clean single-target ChEMBL pIC50 dataset.

This version uses the public ChEMBL API instead of downloading the full
multi-gigabyte SQLite dump. That keeps the workshop project simple while still
producing a real dataset with pIC50, SMILES and RDKit descriptors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from tqdm import tqdm


API_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
DEFAULT_TARGET = "CHEMBL203"  # EGFR, a common single-protein drug-discovery target.
DEFAULT_OUTPUT = Path("data/processed/clean_target_dataset.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"ChEMBL target id. Defaults to {DEFAULT_TARGET} (EGFR).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=10000,
        help="Cap output size for a simple workshop-sized dataset. Use 0 for no cap.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Defaults to {DEFAULT_OUTPUT}.",
    )
    return parser.parse_args()


def fetch_json(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}/{endpoint}",
        params=params,
        timeout=60,
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    return response.json()


def fetch_target(target_chembl_id: str) -> dict[str, Any]:
    return fetch_json(f"target/{target_chembl_id}.json")


def fetch_activities(target_chembl_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    limit = 1000
    offset = 0

    with tqdm(desc="Downloading ChEMBL activities", unit="page") as progress:
        while True:
            payload = fetch_json(
                "activity.json",
                params={
                    "target_chembl_id": target_chembl_id,
                    "standard_type": "IC50",
                    "limit": limit,
                    "offset": offset,
                },
            )
            page_rows = payload.get("activities", [])
            rows.extend(page_rows)
            progress.update(1)

            if len(page_rows) < limit:
                break
            offset += limit

    return pd.DataFrame(rows)


def is_not_duplicate(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {"0", "false", "none", ""}


def clean_measurements(raw: pd.DataFrame, target: dict[str, Any]) -> pd.DataFrame:
    if raw.empty:
        return raw

    df = raw.copy()
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")

    required_columns = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_relation",
        "standard_units",
        "pchembl_value",
    ]
    df = df.dropna(subset=required_columns)
    df = df[
        (df["standard_relation"] == "=")
        & (df["standard_units"] == "nM")
        & (df["pchembl_value"].between(2.0, 14.0))
        & (df["potential_duplicate"].apply(is_not_duplicate))
    ].copy()

    df["target_chembl_id"] = target.get("target_chembl_id")
    df["pref_name"] = target.get("pref_name")
    df["organism"] = target.get("organism")
    df["target_type"] = target.get("target_type")
    return df


def aggregate_measurements(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["molecule_chembl_id", "canonical_smiles"], as_index=False)
        .agg(
            pIC50=("pchembl_value", "median"),
            pIC50_mean=("pchembl_value", "mean"),
            pIC50_std=("pchembl_value", "std"),
            measurement_count=("pchembl_value", "size"),
            standard_value_nM=("standard_value", "median"),
            target_chembl_id=("target_chembl_id", "first"),
            pref_name=("pref_name", "first"),
            organism=("organism", "first"),
            target_type=("target_type", "first"),
            assay_type=("assay_type", lambda values: ",".join(sorted(set(values.dropna())))),
        )
    )
    grouped["pIC50_std"] = grouped["pIC50_std"].fillna(0.0)
    return grouped


def descriptors_from_smiles(smiles: str) -> dict[str, float] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
    }


def add_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas(desc="RDKit descriptors")
    descriptors = df["canonical_smiles"].progress_apply(descriptors_from_smiles)
    valid_mask = descriptors.notna()
    invalid_count = int((~valid_mask).sum())
    if invalid_count:
        print(f"Dropping {invalid_count} rows with invalid SMILES.")

    descriptor_df = pd.DataFrame(descriptors[valid_mask].tolist(), index=df.index[valid_mask])
    return pd.concat([df.loc[valid_mask].reset_index(drop=True), descriptor_df.reset_index(drop=True)], axis=1)


def cap_records(df: pd.DataFrame, max_records: int) -> pd.DataFrame:
    if max_records <= 0 or len(df) <= max_records:
        return df

    # Deterministic sample keeps the dataset workshop-sized without changing its distribution too much.
    return (
        df.sample(n=max_records, random_state=42)
        .sort_values(["molecule_chembl_id"])
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()

    target = fetch_target(args.target)
    print(
        "Target: "
        f"{target.get('target_chembl_id')} - {target.get('pref_name')} "
        f"({target.get('organism')}, {target.get('target_type')})"
    )

    raw = fetch_activities(args.target)
    if raw.empty:
        raise RuntimeError(f"No IC50 activities found for target={args.target}.")

    print(f"\nRaw IC50 activities: {len(raw):,}")
    clean = clean_measurements(raw, target)
    print(f"Clean exact IC50 nM activities: {len(clean):,}")
    if clean.empty:
        raise RuntimeError("No rows left after cleaning.")

    dataset = aggregate_measurements(clean)
    print(f"Unique molecules after aggregation: {len(dataset):,}")

    dataset = add_descriptors(dataset)
    dataset = cap_records(dataset, args.max_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print(f"\nSaved dataset: {args.output}")
    print(f"Rows: {len(dataset):,}")
    print(f"Target: {dataset['pref_name'].iloc[0]} ({dataset['organism'].iloc[0]})")
    print("\npIC50 summary:")
    print(dataset["pIC50"].describe().round(3).to_string())


if __name__ == "__main__":
    main()
