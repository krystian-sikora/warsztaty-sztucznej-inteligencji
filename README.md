# pIC50 dla EGFR — warsztaty ML

Projekt regresji przewidujący wartość **pIC50** cząsteczek chemicznych wobec jednego targetu: **EGFR (CHEMBL203)**. Pipeline obejmuje pobranie danych z ChEMBL, czyszczenie, EDA, podziały train/val/test, baseline MLP, model grafowy GCN, strojenie hiperparametrów oraz inferencję ze SMILES z opcjonalnym asystentem LLM.

## Pipeline

```text
ChEMBL API
    → prepare_dataset.py   (czyszczenie, deskryptory RDKit)
    → target_eda.py        (EDA)
    → create_splits.py     (random + scaffold, 80/10/10)
    → train_baseline.py    (MLP na deskryptorach)
    → train_gnn.py         (GCN ze SMILES)
    → tune_*.py            (strojenie na random val)
    → predict_pic50.py     (inferencja + narzędzie dla LLM)
    → app.py               (Streamlit: wyniki + asystent)
```

## Finalne funkcjonalności

- **Dataset EGFR** — ok. 10 000 rekordów po czyszczeniu (`canonical_smiles`, `pIC50`, deskryptory RDKit).
- **Dwa podziały danych** — `random` (główny do strojenia) i `scaffold` (trudniejsza generalizacja chemiczna).
- **Baseline MLP** i **GCN** z porównaniem metryk (MAE, RMSE, R²).
- **Strojenie hiperparametrów** — wybór najlepszej konfiguracji po `random_val R2`.
- **Predykcja pIC50 ze SMILES** — skrypt CLI i moduł Pythona (`scripts/predict_pic50.py`, `scripts/pic50_inference.py`).
- **Asystent w Streamlit** — podsumowanie projektu, metryki, chat z opcjonalnym wywołaniem narzędzia predykcji (OpenCode Zen, OpenAI, Ollama z narzędziami).
- **Logowanie wywołań narzędzia** — `logs/pic50_tool.jsonl` (walidacja, czy LLM faktycznie użył predykcji).

## Przygotowanie danych

Skrypt: `scripts/prepare_dataset.py`

- Źródło: publiczne **API ChEMBL** (bez pobierania pełnej bazy SQLite).
- Tylko pomiary **IC50** w **nM**, relacja `=`, z `pchembl_value` i `canonical_smiles`.
- Agregacja powtórzeń tej samej cząsteczki — **mediana pIC50**.
- Walidacja SMILES przez **RDKit**; obliczenie deskryptorów: `MW`, `LogP`, `TPSA`, `HBD`, `HBA`, `rotatable_bonds`, `heavy_atoms`.

Wynik: `data/processed/clean_target_dataset.csv`

## Podział danych

Skrypt: `scripts/create_splits.py`

| Wariant   | Opis |
|-----------|------|
| **random**   | Losowy podział 8000 / 1000 / 1000 (train / val / test) — używany do strojenia i końcowej oceny. |
| **scaffold** | Podział według szkieletu chemicznego (Murcko) — trudniejszy test generalizacji. |

Pliki: `data/splits/{random,scaffold}_{train,val,test}.csv`

## Finalny model

**Dostrojony GCN** (PyTorch, bez PyG/DGL) — wybrany po najlepszym `random_val R2`.

| Parametr | Wartość |
|----------|---------|
| `hidden_dim` | 128 |
| `num_layers` | 3 |
| `dropout` | 0.1 |
| `learning_rate` | 0.001 |

**Metryki na random test:**

| Metryka | Wartość |
|---------|---------|
| R² | 0.5242 |
| MAE | 0.7152 |
| RMSE | 0.9015 |

Wagi i konfiguracja: `reports/tuning/gnn/best_model.pt`, `reports/tuning/gnn/best_config.json`

Wejście inferencji: **canonical SMILES** → graf molekularny (atomy, wiązania, cechy atomów) → przewidywana **pIC50**.

> Wynik R² ≈ 0.52 dotyczy **random test split**. Model nie jest przeznaczony do realnego odkrywania leków — to pipeline warsztatowy.

## Struktura projektu

```text
warsztaty-sztucznej-inteligencji/
├── app.py                      # Streamlit UI + asystent LLM
├── llm_assistant.py            # tool calling (predict_pic50)
├── requirements.txt
├── agent/                      # dokumentacja kroków (PL)
│   ├── podsumowanie-pracy.md
│   └── narzedzie-predykcji-pic50.md
├── scripts/
│   ├── prepare_dataset.py      # ChEMBL → CSV
│   ├── target_eda.py
│   ├── create_splits.py
│   ├── train_baseline.py
│   ├── train_gnn.py
│   ├── tune_baseline.py
│   ├── tune_gnn.py
│   ├── export_gcn_weights.py   # generowanie best_model.pt
│   ├── pic50_inference.py      # logika inferencji
│   ├── predict_pic50.py        # CLI dla agentów / LLM
│   └── pic50_tool_logging.py
├── data/
│   ├── processed/              # clean_target_dataset.csv (po prepare)
│   └── splits/                 # random_*, scaffold_* (po create_splits)
├── reports/                    # metryki, wykresy, tuning
├── logs/                       # pic50_tool.jsonl (gitignore)
└── EDA.ipynb
```

## Instalacja i setup

### Wymagania

- Python 3.10+
- Windows / Linux / macOS

### 1. Klonowanie i środowisko wirtualne

```powershell
git clone <url-repozytorium>
cd warsztaty-sztucznej-inteligencji
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Dane (opcjonalnie — jeśli brak lokalnych CSV)

```powershell
python scripts\prepare_dataset.py --target CHEMBL203
python scripts\create_splits.py
```

### 3. Wagi modelu do predykcji

Jeśli brakuje `reports/tuning/gnn/best_model.pt`:

```powershell
python scripts\export_gcn_weights.py
```

(albo pełne strojenie: `python scripts\tune_gnn.py`)

### 4. Predykcja ze SMILES

```powershell
python scripts\predict_pic50.py --smiles "CCO" --pretty
```

### 5. Interfejs Streamlit

```powershell
streamlit run app.py
```

W panelu bocznym wybierz backend asystenta (np. **OpenCode Zen** z kluczem API). W trybach z narzędziami model może wywołać lokalny GCN i zwrócić przewidywaną pIC50.

### 6. Pełny pipeline ML (od zera)

```powershell
python scripts\prepare_dataset.py
python scripts\target_eda.py
python scripts\create_splits.py
python scripts\train_baseline.py
python scripts\train_gnn.py
python scripts\tune_baseline.py
python scripts\tune_gnn.py
```

## Dokumentacja szczegółowa

- [agent/podsumowanie-pracy.md](agent/podsumowanie-pracy.md) — pełne podsumowanie wyników
- [agent/narzedzie-predykcji-pic50.md](agent/narzedzie-predykcji-pic50.md) — kontrakt narzędzia predykcji i logowanie
- [agent/krok-*.md](agent/) — opisy poszczególnych kroków warsztatu
