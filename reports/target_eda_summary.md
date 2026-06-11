# EDA targetu EGFR

## Zakres analizy

Analiza dotyczy przygotowanego datasetu:

```text
data/processed/clean_target_dataset.csv
```

Target:

```text
Epidermal growth factor receptor
Homo sapiens
```

Liczba rekordów: `10,000`  
Liczba unikalnych identyfikatorów cząsteczek (`standard_inchi_key`): `10,000`  
Liczba zduplikowanych `canonical_smiles`: `0`  
Liczba cząsteczek z więcej niż jednym pomiarem: `3,315`

## Rozkład pIC50

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

Interpretacja: wartości `pIC50` są już w skali logarytmicznej, więc nie trzeba ich dodatkowo transformować przed baseline. Większe `pIC50` oznacza silniejszą aktywność wobec targetu.

Wykres:

```text
reports\figures\pIC50_distribution.png
```

## Deskryptory fizykochemiczne

W zbiorze znajdują się podstawowe deskryptory RDKit: `MW`, `LogP`, `TPSA`, `HBD`, `HBA`, `rotatable_bonds`, `heavy_atoms`.

Statystyki deskryptorów:

```text
              MW       LogP       TPSA        HBD        HBA  rotatable_bonds  heavy_atoms
count  10000.000  10000.000  10000.000  10000.000  10000.000        10000.000    10000.000
mean     481.455      4.589     95.988      2.206      6.716            6.930       33.918
std      119.656      1.431     30.435      1.180      2.158            3.209        8.571
min      110.112     -5.994      0.000      0.000      0.000            0.000        3.000
25%      401.469      3.672     76.830      1.000      5.000            5.000       28.000
50%      480.366      4.542     96.370      2.000      7.000            7.000       34.000
75%      554.681      5.483    111.443      3.000      8.000            9.000       39.000
max     1425.805     13.029    530.870     17.000     19.000           36.000      103.000
```

Wykres:

```text
reports\figures\descriptor_distributions.png
```

## Korelacje z pIC50

Najsilniejsze bezwzględne korelacje Spearmana z `pIC50` mają:

```text
MW, heavy_atoms, HBA
```

To nie oznacza jeszcze przyczynowości ani dobrego modelu, ale pomaga zobaczyć, które proste cechy mogą być przydatne dla baseline.

Wykres:

```text
reports\figures\descriptor_correlations.png
```

## Liczba pomiarów

Kolumna `measurement_count` pokazuje, ile pomiarów źródłowych zostało zagregowanych do jednego rekordu cząsteczki. Większość rekordów ma pojedynczy pomiar, więc `pIC50_std` często wynosi `0`.

Wykres:

```text
reports\figures\measurement_count.png
```

## Wnioski praktyczne

- Dataset jest wystarczająco mały, żeby trenować prosty baseline bez dużej infrastruktury.
- Dane dotyczą jednego targetu, więc unikamy najważniejszego błędu z poprzedniego EDA: mieszania wielu białek w jednym problemie regresji.
- Najbliższy sensowny krok to przygotowanie splitów `train/validation/test` oraz prostego baseline na deskryptorach RDKit.
