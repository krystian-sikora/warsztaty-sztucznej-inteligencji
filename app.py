"""Simple Streamlit UI for the pIC50 project summary and text-only assistant."""

from __future__ import annotations

import json
from pathlib import Path

import requests
import streamlit as st


PROJECT_SUMMARY_PATH = Path("agent/podsumowanie-pracy.md")
BEST_GCN_CONFIG_PATH = Path("reports/tuning/gnn/best_config.json")
BEST_MLP_CONFIG_PATH = Path("reports/tuning/baseline/best_config.json")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


DEFAULT_PROJECT_CONTEXT = """Projekt przewiduje pIC50 dla cząsteczek wobec targetu EGFR.
Najlepszy wynik uzyskał dostrojony GCN: random_test R2 = 0.5242.
MLP na deskryptorach fizykochemicznych był słabszy.
Random split był głównym podziałem do strojenia, a scaffold split pozostaje trudniejszą oceną generalizacji.
"""


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_project_context() -> tuple[str, dict[str, object], dict[str, object]]:
    summary = read_text(PROJECT_SUMMARY_PATH) or DEFAULT_PROJECT_CONTEXT
    gcn_config = read_json(BEST_GCN_CONFIG_PATH)
    mlp_config = read_json(BEST_MLP_CONFIG_PATH)
    return summary, gcn_config, mlp_config


def format_metric(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return "brak danych"


def generate_text_response(question: str, project_context: str) -> str:
    """Return a deterministic text-only answer based on local project context."""
    normalized = question.lower().strip()
    if not normalized:
        return "Wpisz pytanie o projekt, modele, metryki albo ograniczenia wyniku."

    if any(keyword in normalized for keyword in ["pic50", "ic50", "co przewiduje", "target"]):
        return (
            "Projekt przewiduje wartość pIC50 dla cząsteczek chemicznych wobec jednego targetu biologicznego: "
            "EGFR (CHEMBL203). Jest to problem regresji, czyli model zwraca liczbę opisującą aktywność związku."
        )

    if any(keyword in normalized for keyword in ["najlepszy", "wynik", "r2", "metryk"]):
        return (
            "Najlepszy wynik uzyskał dostrojony model GCN. Został wybrany po random_val R2 = 0.4651, "
            "a na random_test osiągnął R2 = 0.5242, MAE = 0.7152 i RMSE = 0.9015."
        )

    if any(keyword in normalized for keyword in ["mlp", "baseline", "deskryptor"]):
        return (
            "Baseline MLP korzystał tylko z deskryptorów RDKit: MW, LogP, TPSA, HBD, HBA, rotatable_bonds "
            "i heavy_atoms. Po strojeniu najlepszy MLP osiągnął random_test R2 = 0.3429, więc był słabszy od GCN."
        )

    if any(keyword in normalized for keyword in ["gcn", "gnn", "graf", "smiles"]):
        return (
            "GCN korzystał ze struktury cząsteczki zapisanej jako SMILES. Skrypt zamieniał cząsteczkę na graf, "
            "gdzie atomy były węzłami, a wiązania krawędziami. Najlepsza konfiguracja miała hidden_dim = 128, "
            "3 warstwy GCN, dropout = 0.1 i learning_rate = 0.001."
        )

    if any(keyword in normalized for keyword in ["random", "scaffold", "split", "podział"]):
        return (
            "Random split był głównym podziałem do strojenia i końcowej oceny najlepszego wyniku. Scaffold split "
            "jest trudniejszy, bo rozdziela cząsteczki według szkieletów chemicznych i lepiej sprawdza generalizację."
        )

    if any(keyword in normalized for keyword in ["ogranic", "limit", "problem", "ryzyko"]):
        return (
            "Najważniejsze ograniczenie jest takie, że wynik R2 = 0.5242 dotyczy random test split, czyli łatwiejszej "
            "ewaluacji. Model nie jest gotowy do realnego odkrywania leków, ale pokazuje poprawny pipeline projektu."
        )

    if any(keyword in normalized for keyword in ["dane", "dataset", "chembl", "egfr"]):
        return (
            "Dataset pochodzi z ChEMBL API i dotyczy targetu EGFR (CHEMBL203). Po czyszczeniu zawiera 10 000 "
            "rekordów, bez zduplikowanych canonical_smiles, z pIC50 jako medianą powtórzonych pomiarów cząsteczki."
        )

    context_hint = " ".join(project_context.split()[:40])
    return (
        "Nie mam wystarczającego kontekstu w tej prostej wersji UI, żeby odpowiedzieć szczegółowo. "
        f"Mogę odpowiadać głównie o danych, pIC50, splitach, MLP, GCN, metrykach i ograniczeniach. Kontekst projektu: {context_hint}..."
    )


def build_ollama_prompt(question: str, project_context: str) -> str:
    short_context = project_context[:6000]
    return f"""Jesteś prostym asystentem tekstowym dla projektu ML przewidującego pIC50 dla EGFR.
Odpowiadasz wyłącznie tekstem.
Nie możesz wywoływać narzędzi, uruchamiać kodu, pobierać danych, trenować modeli ani modyfikować plików.
Jeśli pytanie wykracza poza kontekst projektu, powiedz krótko, że nie masz wystarczającego kontekstu.

Kontekst projektu:
{short_context}

Pytanie użytkownika:
{question}

Odpowiedź po polsku:"""


def generate_ollama_response(
    question: str,
    project_context: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
) -> str:
    """Call local Ollama and return text only, with a safe fallback message."""
    if not question.strip():
        return "Wpisz pytanie o projekt, modele, metryki albo ograniczenia wyniku."

    payload = {
        "model": model,
        "prompt": build_ollama_prompt(question, project_context),
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 350,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return (
            "Nie mogę połączyć się z lokalną Ollama. Uruchom Ollama i pobierz mały model, np.:\n\n"
            "```powershell\n"
            "ollama pull llama3.2:3b\n"
            "ollama serve\n"
            "```\n\n"
            "Możesz też przełączyć backend na lokalny stub."
        )
    except requests.exceptions.HTTPError as error:
        return (
            f"Ollama zwróciła błąd HTTP: {error}. Sprawdź, czy model `{model}` jest pobrany, np. "
            f"`ollama pull {model}`."
        )
    except requests.exceptions.RequestException as error:
        return f"Nie udało się uzyskać odpowiedzi z Ollama: {error}"

    data = response.json()
    text = str(data.get("response", "")).strip()
    if not text:
        return "Ollama odpowiedziała pustym tekstem. Spróbuj ponownie albo wybierz inny model."
    return text


def render_metric_cards(gcn_config: dict[str, object], mlp_config: dict[str, object]) -> None:
    gcn_col, mlp_col = st.columns(2)
    with gcn_col:
        st.subheader("Najlepszy GCN")
        st.metric("random_test R2", format_metric(gcn_config.get("test_R2")))
        st.write(
            {
                "random_val R2": format_metric(gcn_config.get("val_R2")),
                "MAE": format_metric(gcn_config.get("test_MAE")),
                "RMSE": format_metric(gcn_config.get("test_RMSE")),
            }
        )
    with mlp_col:
        st.subheader("Najlepszy MLP")
        st.metric("random_test R2", format_metric(mlp_config.get("test_R2")))
        st.write(
            {
                "random_val R2": format_metric(mlp_config.get("val_R2")),
                "MAE": format_metric(mlp_config.get("test_MAE")),
                "RMSE": format_metric(mlp_config.get("test_RMSE")),
            }
        )


def main() -> None:
    st.set_page_config(page_title="pIC50 Project UI", layout="wide")

    project_context, gcn_config, mlp_config = load_project_context()

    st.sidebar.header("Ustawienia asystenta")
    backend = st.sidebar.radio(
        "Backend odpowiedzi",
        ["Ollama lokalnie", "Lokalny stub"],
        help="Ollama używa lokalnego modelu LLM. Stub działa bez żadnego modelu.",
    )
    ollama_model = st.sidebar.text_input("Model Ollama", value=DEFAULT_OLLAMA_MODEL)

    with st.sidebar.expander("Jak uruchomić Ollama"):
        st.code(
            "ollama pull llama3.2:3b\nollama serve",
            language="powershell",
        )
        st.write("Jeśli Ollama działa w tle jako aplikacja desktopowa, osobne `ollama serve` może nie być potrzebne.")

    st.title("pIC50 dla EGFR - prosty interfejs projektu")
    st.write(
        "Ten interfejs pokazuje najważniejsze wyniki projektu i udostępnia prostego asystenta tekstowego. "
        "Asystent odpowiada tekstem, nie wywołuje narzędzi i nie uruchamia treningu."
    )

    st.info("Najlepszy model: tuned GCN, random_test R2 = 0.5242.")
    render_metric_cards(gcn_config, mlp_config)

    with st.expander("Ograniczenia wyniku"):
        st.write(
            "Wynik R2 = 0.5242 dotyczy random test split. Scaffold split pozostaje trudniejszą i bardziej "
            "realistyczną oceną generalizacji do nowych rodzin chemicznych."
        )

    with st.expander("Pełne podsumowanie pracy"):
        st.markdown(project_context)

    st.header("Lokalny asystent tekstowy")
    question = st.text_area(
        "Zadaj pytanie o projekt",
        placeholder="Np. Jaki model uzyskał najlepszy wynik? Czym różni się MLP od GCN?",
        height=100,
    )

    if st.button("Odpowiedz"):
        if backend == "Ollama lokalnie":
            response = generate_ollama_response(question, project_context, model=ollama_model)
        else:
            response = generate_text_response(question, project_context)
        st.markdown(response)


if __name__ == "__main__":
    main()
