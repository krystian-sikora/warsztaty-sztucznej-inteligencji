# Krok 6: strojenie parametrów pod cel R2 > 0.5

## Cel

Celem tego kroku było sprawdzenie, czy da się osiągnąć `R2 > 0.5` na `random test split` bez dodawania nowych cech, czyli wyłącznie przez strojenie parametrów istniejących modeli:

- baseline MLP na cechach fizykochemicznych RDKit,
- prosty GCN na grafach cząsteczek tworzonych ze `SMILES`.

Model wybierano po wyniku `random_val R2`. `random_test R2` służył dopiero do końcowej oceny najlepszego wariantu.

## Wykonane skrypty

Dodano dwa skrypty tuningowe:

```text
scripts/tune_baseline.py
scripts/tune_gnn.py
```

Artefakty zapisano w:

```text
reports/tuning/baseline/
reports/tuning/gnn/
```

Każdy katalog zawiera m.in.:

- `metrics.csv` - wyniki wszystkich konfiguracji,
- `best_config.json` - najlepszą konfigurację wybraną po `random_val R2`,
- `best_loss_history.csv` i `best_loss.png` - krzywą uczenia najlepszego wariantu,
- `best_test_predictions.png` - wykres predykcji dla `random_test`.

## Tuning MLP

Najlepszy wariant MLP:

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

Wniosek: samo strojenie MLP na obecnych deskryptorach fizykochemicznych poprawiło wynik tylko nieznacznie względem wcześniejszego baseline i nie wystarczyło do osiągnięcia `R2 > 0.5`.

## Tuning GCN

Najlepszy wariant GCN:

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

Wniosek: najlepszy GCN został wybrany po najwyższym `random_val R2` i przekroczył wymagany próg na `random_test`.

## Końcowa ocena celu

Cel `R2 > 0.5` na `random test split` został osiągnięty przez strojony GCN:

```text
best model: tuned GCN
selection metric: random_val R2 = 0.4651
final random_test R2 = 0.5242
target threshold: 0.5000
result: achieved
```

Wynik jest zgodny z nowym celem projektu, ale trzeba pamiętać, że dotyczy on `random split`. Dla bardziej wymagającej oceny uogólniania nadal ważny pozostaje `scaffold split`, który wcześniej dawał niższe wyniki.
