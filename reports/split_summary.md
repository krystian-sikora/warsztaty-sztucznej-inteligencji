# Krok 3: podziały danych

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

Liczba rekordów wejściowych: `10,000`  
Kolumna identyfikująca cząsteczkę: `standard_inchi_key`

## Random split

Pliki:

```text
data\splits\random_train.csv
data\splits\random_val.csv
data\splits\random_test.csv
```

Statystyki:

```text
split  rows  unique_molecules  pIC50_mean  pIC50_std  pIC50_min  pIC50_max
train  8000              8000       6.920      1.303        4.0      10.60
  val  1000              1000       6.915      1.325        4.0      11.00
 test  1000              1000       6.884      1.308        4.0      10.54
```

## Scaffold split

Pliki:

```text
data\splits\scaffold_train.csv
data\splits\scaffold_val.csv
data\splits\scaffold_test.csv
```

Statystyki:

```text
split  rows  unique_molecules  pIC50_mean  pIC50_std  pIC50_min  pIC50_max  unique_scaffolds
train  8000              8000       6.960      1.311        4.0      10.60              1598
  val  1000              1000       6.716      1.282        4.0      11.00              1000
 test  1000              1000       6.765      1.254        4.0      10.15              1000
```

## Kontrola przecieków

W obu wariantach sprawdzono, że identyfikatory cząsteczek nie powtarzają się między `train`, `validation` i `test`.

Dla `scaffold split` dodatkowo sprawdzono, że te same wartości `scaffold` nie występują w wielu zbiorach.

## Wniosek

Podziały są gotowe do kolejnego kroku: treningu prostego baseline na deskryptorach RDKit.
