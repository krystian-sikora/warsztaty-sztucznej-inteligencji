# Krok 1: przygotowanie środowiska i datasetu

## Co zostało wykonane

W tym kroku przygotowano minimalne środowisko oraz pierwszy czysty dataset do projektu przewidywania `pIC50` dla jednego targetu białkowego.

Utworzono wirtualne środowisko Pythona:

```powershell
python -m venv .venv
```

Zależności zapisano w pliku [`requirements.txt`](../requirements.txt). Aktualnie projekt używa tylko lekkiego zestawu bibliotek:

- `numpy`,
- `pandas`,
- `rdkit`,
- `requests`,
- `tqdm`.

Początkowo rozważane było pobieranie pełnej bazy ChEMBL SQLite, ale ten wariant okazał się zbyt ciężki dla prostego projektu, bo archiwum ChEMBL ma kilka GB. Dlatego przygotowano prostszy skrypt oparty o publiczne API ChEMBL.

Główny skrypt znajduje się w [`scripts/prepare_dataset.py`](../scripts/prepare_dataset.py).

Domyślnie skrypt przygotowuje dane dla targetu:

```text
CHEMBL203 - Epidermal growth factor receptor (EGFR)
organizm: Homo sapiens
typ targetu: SINGLE PROTEIN
```

Dataset został zapisany do:

```text
data/processed/clean_target_dataset.csv
```

Wygenerowany zbiór ma rozmiar ograniczony do około 10 000 cząsteczek, żeby projekt pozostał prosty i możliwy do dalszego trenowania bez dużej infrastruktury.

## Jak uruchomić skrypt

Po aktywacji środowiska:

```powershell
.\.venv\Scripts\Activate.ps1
```

można ponownie wygenerować dataset poleceniem:

```powershell
python scripts\prepare_dataset.py --target CHEMBL203 --max-records 10000
```

Parametr `--target` pozwala wskazać inny target ChEMBL, a `--max-records` ogranicza rozmiar wynikowego pliku. Wartość `0` oznacza brak limitu.

## Jak skrypt pobiera dane

Skrypt nie pobiera całej bazy ChEMBL. Zamiast tego odpytuje publiczne API:

```text
https://www.ebi.ac.uk/chembl/api/data
```

Najpierw pobierane są metadane targetu, a potem aktywności biologiczne dla wskazanego targetu z filtrem:

```text
standard_type = IC50
```

Dane są pobierane stronicami po 1000 rekordów, aż API zwróci ostatnią stronę.

## Jak skrypt czyści dane

Czyszczenie danych odbywa się w funkcji `clean_measurements`.

Najpierw wartości `pchembl_value` i `standard_value` są konwertowane na liczby. Rekordy, których nie da się poprawnie skonwertować, są później usuwane przez filtr braków danych.

Następnie skrypt usuwa rekordy z brakami w kluczowych kolumnach:

- `molecule_chembl_id`,
- `canonical_smiles`,
- `standard_relation`,
- `standard_units`,
- `pchembl_value`.

Po usunięciu braków zostają tylko rekordy spełniające wszystkie warunki:

- `standard_relation == "="`, czyli zostają tylko dokładne pomiary, bez wartości typu `>`, `<`, `>=`;
- `standard_units == "nM"`, czyli wszystkie pomiary są w tej samej jednostce;
- `pchembl_value` mieści się w zakresie od `2.0` do `14.0`, co usuwa skrajnie podejrzane wartości;
- `potential_duplicate` nie wskazuje duplikatu.

Do danych dopisywane są też informacje o targetcie:

- `target_chembl_id`,
- `pref_name`,
- `organism`,
- `target_type`.

## Agregacja powtórzonych pomiarów

Po czyszczeniu skrypt agreguje pomiary na poziomie jednej cząsteczki i jednego zapisu `SMILES`.

Grupowanie odbywa się po:

```text
molecule_chembl_id
canonical_smiles
```

Dzięki temu wiele pomiarów tej samej cząsteczki wobec EGFR zostaje sprowadzone do jednego rekordu. Jako główną etykietę regresji skrypt zapisuje:

```text
pIC50 = mediana pchembl_value
```

Dodatkowo zachowywane są:

- `pIC50_mean`,
- `pIC50_std`,
- `measurement_count`,
- `standard_value_nM`.

Jeśli cząsteczka ma tylko jeden pomiar, `pIC50_std` jest ustawiane na `0.0`.

## Deskryptory RDKit

Dla każdego poprawnego `canonical_smiles` skrypt próbuje utworzyć cząsteczkę RDKit.

Jeśli RDKit nie potrafi sparsować danego `SMILES`, rekord jest usuwany. To jest ważne, bo późniejszy model GNN również będzie wymagał poprawnej struktury cząsteczki.

Dla poprawnych cząsteczek liczone są podstawowe cechy fizykochemiczne:

- `MW` - masa cząsteczkowa,
- `LogP` - lipofilowość,
- `TPSA` - topologiczna polarna powierzchnia,
- `HBD` - liczba donorów wiązań wodorowych,
- `HBA` - liczba akceptorów wiązań wodorowych,
- `rotatable_bonds` - liczba rotowalnych wiązań,
- `heavy_atoms` - liczba ciężkich atomów.

Te cechy będą użyte w kolejnym kroku jako prosty baseline dla modelu regresyjnego.

## Wynik kroku

Po tym kroku projekt ma:

- działające środowisko `.venv`,
- listę zależności w [`requirements.txt`](../requirements.txt),
- skrypt przygotowania danych w [`scripts/prepare_dataset.py`](../scripts/prepare_dataset.py),
- pierwszy dataset dla jednego targetu w [`data/processed/clean_target_dataset.csv`](../data/processed/clean_target_dataset.csv).

Dataset jest gotowy do następnego kroku: EDA specyficznego dla EGFR oraz przygotowania prostego baseline modelu przewidującego `pIC50`.
