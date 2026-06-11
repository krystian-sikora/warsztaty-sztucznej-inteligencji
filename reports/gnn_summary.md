# Krok 5: prosty model GCN na grafach cząsteczek

## Cel

Celem tego kroku było wytrenowanie pierwszego modelu grafowego na strukturach cząsteczek zapisanych jako `SMILES`.

Model nie używa jeszcze PyTorch Geometric. Dla prostoty projektowej zaimplementowano mały GCN w czystym PyTorch:

```text
GCN layers: 2
hidden_dim: 64
epochs: 60
batch_size: 64
learning_rate: 0.001
```

## Wejście modelu

Dla każdego `SMILES` tworzony jest graf:

- atomy są węzłami,
- wiązania są krawędziami,
- dodawane są self-loopy,
- macierz sąsiedztwa jest normalizowana stopniami węzłów.

Cechy atomów obejmują m.in. typ atomu, stopień, ładunek formalny, liczbę atomów wodoru, aromatyczność i informację o pierścieniu.

## Metryki testowe

```text
split_type split  rows    MAE   RMSE     R2
    random  test  1000 0.8693 1.0882 0.3068
  scaffold  test  1000 0.8611 1.0747 0.2653
```

## Zapisane artefakty

Metryki:

```text
reports\gnn\metrics.csv
reports\gnn\random_metrics.csv
reports\gnn\scaffold_metrics.csv
```

Historia loss:

```text
reports\gnn\loss_history.csv
reports\gnn\random_loss_history.csv
reports\gnn\scaffold_loss_history.csv
```

Wykresy:

```text
reports\gnn\random_loss.png
reports\gnn\scaffold_loss.png
reports\gnn\random_test_predictions.png
reports\gnn\scaffold_test_predictions.png
```

## Wniosek

To jest pierwszy, prosty model grafowy. Jego wynik należy porównać z baseline MLP na deskryptorach RDKit, a nie traktować jako finalny model.
