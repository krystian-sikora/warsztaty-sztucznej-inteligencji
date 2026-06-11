# Krok 4: baseline na cechach fizykochemicznych

## Co zostało wykonane

W tym kroku wytrenowano prosty baseline regresyjny przewidujący `pIC50` na podstawie deskryptorów fizykochemicznych RDKit.

Dodano skrypt:

```text
scripts/train_baseline.py
```

Model został wytrenowany na dwóch wariantach podziału danych:

- `random split`,
- `scaffold split`.

Dzięki temu mamy dwa punkty odniesienia:

- łatwiejszy wariant losowy,
- trudniejszy wariant scaffold, który lepiej sprawdza generalizację do innych rodzin chemicznych.

## Model

Użyty model:

```text
MLPRegressor
hidden_layer_sizes = (64, 32)
epochs = 120
optimizer = Adam
```

Model korzysta wyłącznie z cech:

```text
MW
LogP
TPSA
HBD
HBA
rotatable_bonds
heavy_atoms
```

Przed treningiem cechy są standaryzowane przez `StandardScaler`.

## Jak uruchomić trening

Po aktywacji środowiska:

```powershell
.\.venv\Scripts\Activate.ps1
```

można uruchomić trening poleceniem:

```powershell
python scripts\train_baseline.py --epochs 120
```

## Zapisane artefakty

Wyniki zapisano w katalogu:

```text
reports/baseline
```

Najważniejsze pliki:

```text
reports/baseline/metrics.csv
reports/baseline/metrics.json
reports/baseline/random_metrics.csv
reports/baseline/scaffold_metrics.csv
reports/baseline/random_loss_history.csv
reports/baseline/scaffold_loss_history.csv
reports/baseline/random_loss.png
reports/baseline/scaffold_loss.png
reports/baseline/random_test_predictions.png
reports/baseline/scaffold_test_predictions.png
```

Powstał też raport:

```text
reports/baseline_summary.md
```

## Metryki

Metryki testowe:

```text
split_type split  rows    MAE   RMSE     R2
    random  test  1000 0.8428 1.0737 0.3252
  scaffold  test  1000 0.9519 1.1741 0.1231
```

Interpretacja:

- `random split` daje lepszy wynik, co jest oczekiwane, bo losowy podział jest łatwiejszy;
- `scaffold split` daje słabszy wynik, bo testuje cząsteczki z innych scaffoldów;
- dodatnie `R2` oznacza, że baseline uczy się czegoś lepszego niż samo przewidywanie średniej, ale jakość nadal jest umiarkowana.

## Loss

Skrypt zapisuje historię `train MSE` i `validation MSE` po każdej epoce.

W obu wariantach loss szybko spada na początku, a potem stabilizuje się. Dla scaffold split widać większą różnicę między train i validation, co sugeruje trudniejszą generalizację.

## Wniosek praktyczny

Ten baseline jest punktem odniesienia dla przyszłych modeli.

Jeżeli późniejszy model GNN nie osiągnie lepszych wyników niż:

```text
random test R2: 0.3252
scaffold test R2: 0.1231
```

to sama reprezentacja grafowa nie dała jeszcze praktycznej poprawy w obecnej konfiguracji.

Następny krok może polegać na przygotowaniu reprezentacji grafowej cząsteczek ze `SMILES` albo na krótkim sanity checku/overfittingu modelu na małym podzbiorze.
