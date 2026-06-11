# Krok 4: baseline na cechach fizykochemicznych

## Cel

Celem tego kroku było wytrenowanie prostego modelu regresyjnego przewidującego `pIC50` wyłącznie na podstawie cech fizykochemicznych RDKit.

Model:

```text
MLPRegressor
hidden layers: 64, 32
epochs: 120
features: MW, LogP, TPSA, HBD, HBA, rotatable_bonds, heavy_atoms
target: pIC50
```

Model został wytrenowany osobno dla dwóch wariantów podziału danych:

- `random split`,
- `scaffold split`.

## Metryki testowe

```text
split_type split  rows    MAE   RMSE     R2
    random  test  1000 0.8428 1.0737 0.3252
  scaffold  test  1000 0.9519 1.1741 0.1231
```

Pełne metryki dla `train`, `validation` i `test` zapisano w:

```text
reports\baseline\random_metrics.csv
reports\baseline\scaffold_metrics.csv
```

## Krzywe loss

Historia loss została zapisana w:

```text
reports\baseline\random_loss_history.csv
reports\baseline\scaffold_loss_history.csv
```

Wykresy loss:

```text
reports\baseline\random_loss.png
reports\baseline\scaffold_loss.png
```

## Wykresy predykcji

Wykresy `predykcja pIC50` vs `rzeczywiste pIC50` dla zbiorów testowych:

```text
reports\baseline\random_test_predictions.png
reports\baseline\scaffold_test_predictions.png
```

## Wniosek

Ten baseline jest punktem odniesienia dla przyszłego modelu GNN. Jeżeli GNN nie przebije tego wyniku, to sama reprezentacja grafowa nie dała jeszcze praktycznej poprawy w obecnej konfiguracji.
