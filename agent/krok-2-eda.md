# Krok 2: EDA dla wybranego targetu

## Co zostało wykonane

W tym kroku wykonano proste EDA dla datasetu przygotowanego w kroku 1:

```text
data/processed/clean_target_dataset.csv
```

Analiza dotyczy jednego targetu:

```text
Epidermal growth factor receptor
Homo sapiens
```

Dzięki temu analiza jest zgodna z celem projektu: nie mieszamy wielu różnych białek w jednym problemie regresji.

Dodano skrypt:

```text
scripts/target_eda.py
```

Skrypt czyta przygotowany CSV, liczy podstawowe statystyki i zapisuje raport:

```text
reports/target_eda_summary.md
```

oraz wykresy:

```text
reports/figures/pIC50_distribution.png
reports/figures/descriptor_distributions.png
reports/figures/descriptor_correlations.png
reports/figures/measurement_count.png
```

## Jak uruchomić EDA

Po aktywacji środowiska:

```powershell
.\.venv\Scripts\Activate.ps1
```

można uruchomić analizę poleceniem:

```powershell
python scripts\target_eda.py
```

Domyślnie skrypt używa:

```text
data/processed/clean_target_dataset.csv
```

Można też wskazać własne ścieżki:

```powershell
python scripts\target_eda.py --input data\processed\clean_target_dataset.csv --summary reports\target_eda_summary.md
```

## Zakres analizy

Skrypt sprawdza:

- liczbę rekordów,
- liczbę unikalnych cząsteczek,
- liczbę zduplikowanych `canonical_smiles`,
- liczbę cząsteczek z więcej niż jednym pomiarem,
- rozkład `pIC50`,
- rozkłady deskryptorów RDKit,
- korelacje Spearmana między `pIC50` i deskryptorami,
- rozkład `measurement_count`.

## Najważniejsze wyniki

Dataset ma:

```text
10 000 rekordów
10 000 unikalnych identyfikatorów cząsteczek
0 zduplikowanych canonical_smiles
3 315 cząsteczek z więcej niż jednym pomiarem
```

Podstawowe statystyki `pIC50`:

```text
count    10000.000
mean         6.916
std          1.306
min          4.000
25%          5.960
50%          6.990
75%          7.920
max         11.000
```

Wartości `pIC50` są już w skali logarytmicznej, więc nie trzeba wykonywać dodatkowej transformacji targetu przed baseline.

## Deskryptory

W EDA uwzględniono deskryptory:

- `MW`,
- `LogP`,
- `TPSA`,
- `HBD`,
- `HBA`,
- `rotatable_bonds`,
- `heavy_atoms`.

Najsilniejsze bezwzględne korelacje Spearmana z `pIC50` miały:

```text
MW, heavy_atoms, HBA
```

To jest tylko sygnał eksploracyjny. Nie oznacza jeszcze, że te cechy same wystarczą do dobrego modelowania aktywności, ale są dobrym punktem startowym dla prostego baseline.

## Wnioski praktyczne

- Dataset jest wystarczająco mały i czysty, żeby przejść do prostego baseline.
- Rozkład `pIC50` wygląda sensownie dla problemu regresji.
- Ponieważ dane dotyczą jednego targetu, ewaluacja będzie bardziej uczciwa niż w pierwotnym notebooku EDA mieszającym wiele białek.
- Kolejny krok powinien przygotować podziały `train/validation/test`, najlepiej zaczynając od random split, a potem dodać scaffold split.

## Artefakty po tym kroku

Po tym kroku projekt ma:

- skrypt EDA w [`scripts/target_eda.py`](../scripts/target_eda.py),
- raport EDA w [`reports/target_eda_summary.md`](../reports/target_eda_summary.md),
- wykresy w [`reports/figures`](../reports/figures),
- zaktualizowane zależności w [`requirements.txt`](../requirements.txt).
