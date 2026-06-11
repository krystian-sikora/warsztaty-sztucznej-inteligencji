# Krok 5: prosty model GCN na grafach cząsteczek

## Co zostało wykonane

W tym kroku wytrenowano pierwszy model grafowy przewidujący `pIC50` bezpośrednio ze struktury cząsteczki zapisanej jako `SMILES`.

Dodano skrypt:

```text
scripts/train_gnn.py
```

Model został uruchomiony dla dwóch wariantów podziału danych:

- `random split`,
- `scaffold split`.

## Dlaczego bez PyTorch Geometric

Żeby projekt nie był zbyt skomplikowany, model nie używa PyTorch Geometric ani DGL.

Zamiast tego zaimplementowano mały GCN w czystym PyTorch:

- RDKit parsuje `SMILES`,
- skrypt buduje macierz sąsiedztwa atomów,
- dodawane są self-loopy,
- macierz sąsiedztwa jest normalizowana stopniami węzłów,
- model wykonuje dwie proste warstwy GCN,
- embeddingi atomów są uśredniane do embeddingu całej cząsteczki,
- końcowa głowica liniowa przewiduje `pIC50`.

To jest prosty model grafowy, ale wystarcza jako pierwszy punkt porównania z baseline.

## Cechy atomów

Dla każdego atomu skrypt zapisuje proste cechy:

- typ atomu,
- stopień atomu,
- ładunek formalny,
- liczba atomów wodoru,
- aromatyczność,
- informacja, czy atom jest w pierścieniu.

## Parametry treningu

```text
GCN layers: 2
hidden_dim: 64
epochs: 60
batch_size: 64
learning_rate: 0.001
device: CPU
```

## Jak uruchomić trening

Po aktywacji środowiska:

```powershell
.\.venv\Scripts\Activate.ps1
```

można uruchomić trening:

```powershell
python scripts\train_gnn.py --epochs 60 --batch-size 64
```

## Zapisane artefakty

Wyniki zapisano w:

```text
reports/gnn
```

Najważniejsze pliki:

```text
reports/gnn/metrics.csv
reports/gnn/metrics.json
reports/gnn/random_metrics.csv
reports/gnn/scaffold_metrics.csv
reports/gnn/random_loss_history.csv
reports/gnn/scaffold_loss_history.csv
reports/gnn/random_loss.png
reports/gnn/scaffold_loss.png
reports/gnn/random_test_predictions.png
reports/gnn/scaffold_test_predictions.png
```

Powstał też raport:

```text
reports/gnn_summary.md
```

## Metryki

Metryki testowe GCN:

```text
split_type split  rows    MAE   RMSE     R2
    random  test  1000 0.8693 1.0882 0.3068
  scaffold  test  1000 0.8611 1.0747 0.2653
```

Dla porównania baseline MLP na deskryptorach RDKit miał:

```text
random test R2:   0.3252
scaffold test R2: 0.1231
```

Interpretacja:

- na `random split` GCN jest bardzo blisko baseline, ale minimalnie słabszy;
- na `scaffold split` GCN jest wyraźnie lepszy od baseline;
- to sugeruje, że informacja grafowa może pomagać przy generalizacji do nowych scaffoldów.

## Loss

Loss szybko spada w pierwszych epokach i później stabilizuje się.

Wykresy:

```text
reports/gnn/random_loss.png
reports/gnn/scaffold_loss.png
```

## Wniosek praktyczny

Ten model nie jest jeszcze finalnym GNN, ale spełnia cel projektu jako pierwszy prosty model grafowy:

- korzysta ze struktury cząsteczki,
- działa na obu splitach,
- zapisuje metryki i wykresy,
- daje wynik porównywalny z baseline na random split,
- poprawia wynik na scaffold split.

Kolejny krok powinien zebrać porównanie baseline vs GCN w jednym raporcie i sformułować finalne wnioski.
