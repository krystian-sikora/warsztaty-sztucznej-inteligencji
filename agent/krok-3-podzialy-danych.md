# Krok 3: podziały danych

## Co zostało wykonane

W tym kroku przygotowano podziały danych potrzebne do trenowania i oceniania modeli.

Dodano skrypt:

```text
scripts/create_splits.py
```

Skrypt korzysta z datasetu:

```text
data/processed/clean_target_dataset.csv
```

i tworzy dwa warianty podziału:

- `random split`,
- `scaffold split`.

Oba warianty używają proporcji:

```text
train: 80%
validation: 10%
test: 10%
```

## Dlaczego są dwa podziały

`Random split` jest prostszy i zwykle daje lepsze wyniki. Jest dobry jako pierwszy baseline, bo pozwala szybko sprawdzić, czy pipeline modelowania działa.

`Scaffold split` jest trudniejszy i bardziej realistyczny chemicznie. W tym podziale cząsteczki o tym samym szkielecie chemicznym nie powinny trafiać do różnych zbiorów. Dzięki temu test lepiej sprawdza, czy model generalizuje na nowe rodziny związków, a nie tylko zapamiętuje podobne cząsteczki.

## Jak uruchomić skrypt

Po aktywacji środowiska:

```powershell
.\.venv\Scripts\Activate.ps1
```

można wygenerować splity poleceniem:

```powershell
python scripts\create_splits.py
```

Domyślnie skrypt zapisuje pliki do:

```text
data/splits
```

## Wygenerowane pliki

Random split:

```text
data/splits/random_train.csv
data/splits/random_val.csv
data/splits/random_test.csv
```

Scaffold split:

```text
data/splits/scaffold_train.csv
data/splits/scaffold_val.csv
data/splits/scaffold_test.csv
```

Powstał też raport:

```text
reports/split_summary.md
```

## Jak działa random split

Skrypt losowo miesza cały dataset z ustalonym ziarnem:

```text
seed = 42
```

Następnie bierze:

- pierwsze 80% rekordów jako `train`,
- kolejne 10% jako `validation`,
- ostatnie 10% jako `test`.

Po podziale sprawdzane jest, czy ten sam identyfikator cząsteczki nie występuje w więcej niż jednym zbiorze.

## Jak działa scaffold split

Skrypt wylicza scaffold dla każdej cząsteczki z `canonical_smiles` przy pomocy RDKit:

```text
MurckoScaffold.MurckoScaffoldSmiles
```

Następnie grupuje cząsteczki po scaffoldzie. Cała grupa scaffoldowa trafia tylko do jednego zbioru: `train`, `validation` albo `test`.

Po podziale skrypt sprawdza:

- brak powtórzonych identyfikatorów cząsteczek między zbiorami,
- brak powtórzonych scaffoldów między zbiorami.

## Wyniki podziału

Random split:

```text
train: 8000 rekordów
validation: 1000 rekordów
test: 1000 rekordów
```

Scaffold split:

```text
train: 8000 rekordów
validation: 1000 rekordów
test: 1000 rekordów
```

W raporcie [`reports/split_summary.md`](../reports/split_summary.md) zapisano też średnie, odchylenia i zakresy `pIC50` dla każdego zbioru.

## Wnioski praktyczne

- Splity są gotowe do trenowania baseline.
- Najpierw warto trenować na `random_train.csv` i oceniać na `random_val.csv` oraz `random_test.csv`.
- Potem ten sam model należy sprawdzić na `scaffold split`, bo to będzie trudniejszy i bardziej uczciwy test generalizacji.
- W następnym kroku można zbudować prosty baseline regresyjny na deskryptorach RDKit.

## Artefakty po tym kroku

Po tym kroku projekt ma:

- skrypt podziału danych w [`scripts/create_splits.py`](../scripts/create_splits.py),
- sześć plików splitów w [`data/splits`](../data/splits),
- raport podziałów w [`reports/split_summary.md`](../reports/split_summary.md).
