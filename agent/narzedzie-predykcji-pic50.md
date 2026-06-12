# Narzędzie predykcji pIC50 ze SMILES

## Kiedy używać

Użyj tego narzędzia, gdy użytkownik podaje **canonical SMILES** i chce przewidywaną wartość **pIC50** wobec targetu **EGFR (CHEMBL203)**.

Nie zgaduj wartości — uruchom skrypt predykcji.

## Wymagania wstępne

Wagi modelu muszą istnieć:

```text
reports/tuning/gnn/best_model.pt
reports/tuning/gnn/best_config.json
```

Jeśli brakuje `best_model.pt`, wygeneruj go:

```powershell
.\.venv\Scripts\python.exe scripts\export_gcn_weights.py
```

## Jak wywołać

Z katalogu głównego projektu:

```powershell
# jeden SMILES
.\.venv\Scripts\python.exe scripts\predict_pic50.py --smiles "CCO" --pretty

# wiele SMILES z pliku (jeden na linię)
.\.venv\Scripts\python.exe scripts\predict_pic50.py --input smiles.txt --pretty
```

## Format odpowiedzi (JSON)

```json
{
  "target": "CHEMBL203",
  "target_name": "EGFR",
  "model": "tuned_gcn",
  "predictions": [
    {
      "smiles": "CCO",
      "pic50": 5.1234,
      "status": "ok"
    }
  ],
  "disclaimer": "Wynik dotyczy random-split EGFR (CHEMBL203); model GCN nie jest gotowy do odkrywania leków."
}
```

Pole `status` może być `error` z komunikatem w `message` (np. nieparsowalny SMILES).

## Kody wyjścia

- `0` — wszystkie predykcje OK
- `1` — co najmniej jeden SMILES z błędem
- `2` — brak wag modelu lub konfiguracji

## Ograniczenia

- Tylko target **EGFR (CHEMBL203)**
- Model: dostrojony **GCN** (`random_test R2 ≈ 0.52`)
- SMILES musi być poprawny dla RDKit
- Wynik nie gwarantuje dobrej generalizacji na nowe rodziny chemiczne (scaffold split był trudniejszy)

## Logowanie (walidacja użycia narzędzia)

Każde wywołanie predykcji zapisuje zdarzenia do:

```text
logs/pic50_tool.jsonl
```

Przykładowe zdarzenia:

- `llm_tool_calls` — LLM poprosił o wywołanie narzędzia
- `llm_tool_invoke` — rozpoczęto `predict_pic50` z argumentami
- `prediction_ok` / `prediction_error` — wynik GCN dla SMILES
- `llm_request_done` — koniec odpowiedzi (`tool_used=true/false`)

Podgląd w PowerShell:

```powershell
Get-Content logs\pic50_tool.jsonl -Tail 20
```

Logi trafiają też na stderr (terminal ze Streamlit).

## Moduł Pythona (pod przyszłe API)

Zamiast CLI można zaimportować:

```python
from pic50_inference import predict_pic50, predict_pic50_batch
```

Uruchamiaj z katalogu `scripts/` lub dodaj `scripts/` do `sys.path`.
