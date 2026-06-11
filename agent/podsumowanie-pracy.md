# Podsumowanie pracy nad projektem pIC50

## Cel projektu

Projekt dotyczy przewidywania wartości `pIC50` dla cząsteczek chemicznych wobec jednego targetu biologicznego. Wybrany target to:

```text
CHEMBL203 - Epidermal growth factor receptor (EGFR)
organism: Homo sapiens
target type: SINGLE PROTEIN
```

Problem jest regresją: model dostaje opis cząsteczki, a zwraca przewidywaną wartość `pIC50`. W trakcie pracy przyjęto `R2 > 0.5` na `random test split` jako minimalny próg sukcesu, ale faktycznie strojenie modeli miało prowadzić do możliwie najlepszego wyniku w tej prostej konfiguracji projektu.

## Przygotowanie danych

Zamiast pobierać pełną bazę ChEMBL SQLite, projekt używa publicznego API ChEMBL. Dzięki temu pipeline pozostaje prosty i możliwy do uruchomienia lokalnie bez pobierania kilku GB danych.

Główny skrypt:

```text
scripts/prepare_dataset.py
```

Skrypt pobiera aktywności `IC50` dla jednego targetu, czyści dane i zapisuje gotowy dataset:

```text
data/processed/clean_target_dataset.csv
```

Najważniejsze reguły czyszczenia:

- zostają tylko pomiary `IC50`,
- zostają tylko wartości w `nM`,
- zostają tylko dokładne pomiary `standard_relation == "="`,
- rekord musi mieć `pchembl_value` i `canonical_smiles`,
- usuwane są potencjalne duplikaty,
- `pIC50` musi mieścić się w sensownym zakresie,
- powtórzone pomiary tej samej cząsteczki są agregowane medianą `pIC50`,
- RDKit musi poprawnie sparsować `SMILES`.

Dla każdej cząsteczki policzono też podstawowe deskryptory RDKit:

```text
MW, LogP, TPSA, HBD, HBA, rotatable_bonds, heavy_atoms
```

## EDA

EDA wykonano na przygotowanym datascie dla EGFR.

Skrypt:

```text
scripts/target_eda.py
```

Najważniejsze informacje o danych:

```text
liczba rekordów: 10 000
unikalne identyfikatory cząsteczek: 10 000
zduplikowane canonical_smiles: 0
cząsteczki z więcej niż jednym pomiarem: 3 315
średnie pIC50: 6.916
mediana pIC50: 6.990
```

Najsilniejsze eksploracyjne korelacje Spearmana z `pIC50` miały `MW`, `heavy_atoms` i `HBA`. To potwierdziło, że deskryptory fizykochemiczne są sensownym punktem startowym, ale nie muszą wystarczyć do bardzo dobrego modelu.

Artefakty EDA:

```text
reports/target_eda_summary.md
reports/figures/
```

## Podziały danych

Przygotowano dwa warianty podziału danych:

- `random split`,
- `scaffold split`.

Skrypt:

```text
scripts/create_splits.py
```

Każdy wariant ma proporcje:

```text
train: 8000
validation: 1000
test: 1000
```

`Random split` jest łatwiejszy i został użyty jako główny podział do strojenia oraz końcowej oceny najlepszego wyniku. `Scaffold split` jest trudniejszy chemicznie, bo pilnuje, aby cząsteczki z tym samym szkieletem chemicznym nie trafiały do różnych zbiorów.

## Baseline MLP

Pierwszy baseline używał tylko deskryptorów fizykochemicznych RDKit.

Skrypt:

```text
scripts/train_baseline.py
```

Model:

```text
MLPRegressor
hidden_layer_sizes: 64, 32
epochs: 120
features: MW, LogP, TPSA, HBD, HBA, rotatable_bonds, heavy_atoms
```

Wyniki testowe:

```text
random test R2:   0.3252
random test MAE:  0.8428
random test RMSE: 1.0737

scaffold test R2:   0.1231
scaffold test MAE:  0.9519
scaffold test RMSE: 1.1741
```

Wniosek: baseline uczy się zależności lepiej niż przewidywanie średniej, ale wynik jest umiarkowany i był wyraźnie słabszy od później dostrojonego GCN.

## Pierwszy GCN

Następnie przygotowano prosty model grafowy w czystym PyTorch, bez PyTorch Geometric ani DGL.

Skrypt:

```text
scripts/train_gnn.py
```

Model buduje graf z `SMILES`:

- atomy są węzłami,
- wiązania są krawędziami,
- dodawane są self-loopy,
- macierz sąsiedztwa jest normalizowana,
- cechy atomów obejmują m.in. typ atomu, stopień, ładunek, wodory, aromatyczność i informację o pierścieniu.

Pierwsza konfiguracja:

```text
GCN layers: 2
hidden_dim: 64
epochs: 60
batch_size: 64
learning_rate: 0.001
```

Wyniki testowe:

```text
random test R2:   0.3068
random test MAE:  0.8693
random test RMSE: 1.0882

scaffold test R2:   0.2653
scaffold test MAE:  0.8611
scaffold test RMSE: 1.0747
```

Wniosek: pierwszy GCN był podobny do baseline na `random split`, ale lepszy na `scaffold split`, co sugeruje, że reprezentacja grafowa pomaga w trudniejszej generalizacji chemicznej.

## Strojenie parametrów

Po przyjęciu progu `R2 > 0.5` jako punktu odniesienia sprawdzono strojenie parametrów istniejących modeli, bez dodawania nowych cech typu Morgan fingerprints. Celem praktycznym było nie tylko przekroczenie progu, ale uzyskanie jak najlepszego wyniku w ramach obecnego pipeline'u.

Dodane skrypty:

```text
scripts/tune_baseline.py
scripts/tune_gnn.py
```

Kryterium wyboru modelu:

```text
wybór konfiguracji: random_val R2
końcowa ocena: random_test R2
```

### Najlepszy strojony MLP

```text
hidden_layer_sizes: 128, 64, 32
alpha: 0.0001
learning_rate_init: 0.001
best_epoch: 140
random_val R2: 0.3828
random_test R2: 0.3429
random_test MAE: 0.8038
random_test RMSE: 1.0595
```

Strojenie MLP poprawiło wynik względem pierwszego baseline, ale nadal było wyraźnie słabsze od najlepszego GCN.

### Najlepszy strojony GCN

```text
hidden_dim: 128
num_layers: 3
dropout: 0.1
learning_rate: 0.001
weight_decay: 0.0001
best_epoch: 70
random_train R2: 0.4937
random_val R2: 0.4651
random_test R2: 0.5242
random_test MAE: 0.7152
random_test RMSE: 0.9015
```

Ten wariant został wybrany po najlepszym `random_val R2`, a następnie sprawdzony na `random_test`.

## Końcowy wynik

Minimalny próg `R2 > 0.5` na `random test split` został osiągnięty, a najlepszy wynik uzyskał dostrojony GCN.

```text
najlepszy model: tuned GCN
selection metric: random_val R2 = 0.4651
final random_test R2 = 0.5242
target threshold: 0.5000
status: achieved
```

Najważniejszy wniosek: dla tego datasetu i aktualnych cech najlepszy wynik uzyskał prosty GCN po strojeniu parametrów. Przekroczył on próg `R2 > 0.5` na losowym podziale testowym i był lepszy od wariantów MLP.

## Najważniejsze artefakty

Dokumentacja kroków:

```text
agent/krok-1-przygotowanie-danych.md
agent/krok-2-eda.md
agent/krok-3-podzialy-danych.md
agent/krok-4-baseline.md
agent/krok-5-gnn.md
agent/krok-6-strojenie-r2.md
```

Skrypty:

```text
scripts/prepare_dataset.py
scripts/target_eda.py
scripts/create_splits.py
scripts/train_baseline.py
scripts/train_gnn.py
scripts/tune_baseline.py
scripts/tune_gnn.py
```

Raporty i wyniki:

```text
reports/target_eda_summary.md
reports/split_summary.md
reports/baseline/
reports/gnn/
reports/tuning/baseline/
reports/tuning/gnn/
```

## Ograniczenia

Wynik `R2 = 0.5242` dotyczy `random test split`, czyli łatwiejszego wariantu ewaluacji. `Scaffold split` pozostaje ważny jako trudniejszy i bardziej realistyczny test generalizacji do nowych rodzin chemicznych.

Projekt osiągnął wymagany próg liczbowy i wskazał najlepszy model w ramach przetestowanych konfiguracji, ale nie oznacza to jeszcze modelu gotowego do realnego odkrywania leków. Jest to poprawny, prosty pipeline projektowy: dane z jednego targetu, czyszczenie, EDA, podziały, baseline, GNN, tuning i końcowa ewaluacja.
