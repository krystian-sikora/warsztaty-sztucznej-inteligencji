"""Chat assistants with pIC50 prediction tool (OpenAI-compatible APIs)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI

_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from pic50_inference import WeightsNotFoundError, predict_pic50_batch
from pic50_tool_logging import log_tool_event


MAX_TOOL_ROUNDS = 5

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENCODE_MODEL = "deepseek-v4-flash-free"
DEFAULT_OLLAMA_TOOLS_MODEL = "llama3.1"
OPENCODE_BASE_URL = "https://opencode.ai/zen/v1"
OLLAMA_OPENAI_BASE_URL = "http://localhost:11434/v1"

PREDICT_PIC50_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "predict_pic50",
        "description": (
            "Predict pIC50 for one or more canonical SMILES strings against EGFR (CHEMBL203) "
            "using the project's tuned GCN model."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Canonical SMILES strings to predict.",
                }
            },
            "required": ["smiles"],
        },
    },
}


@dataclass(frozen=True)
class ToolProvider:
    label: str
    base_url: str | None
    default_model: str
    env_var: str | None
    fixed_api_key: str | None = None


TOOL_PROVIDERS: dict[str, ToolProvider] = {
    "opencode": ToolProvider(
        label="OpenCode Zen",
        base_url=OPENCODE_BASE_URL,
        default_model=DEFAULT_OPENCODE_MODEL,
        env_var="OPENCODE_API_KEY",
    ),
    "openai": ToolProvider(
        label="OpenAI",
        base_url=None,
        default_model=DEFAULT_OPENAI_MODEL,
        env_var="OPENAI_API_KEY",
    ),
    "ollama": ToolProvider(
        label="Ollama (lokalnie)",
        base_url=OLLAMA_OPENAI_BASE_URL,
        default_model=DEFAULT_OLLAMA_TOOLS_MODEL,
        env_var=None,
        fixed_api_key="ollama",
    ),
}


def build_system_prompt(project_context: str) -> str:
    short_context = project_context[:6000]
    return f"""Jesteś asystentem projektu ML przewidującego pIC50 dla EGFR (CHEMBL203).

Masz narzędzie predict_pic50. Używaj go, gdy użytkownik podaje canonical SMILES
lub prosi o przewidywaną wartość pIC50 dla cząsteczki. Nie zgaduj wartości liczbowych —
wywołaj narzędzie.

Odpowiadaj po polsku. Po predykcji podaj wynik i przypomnij o ograniczeniach modelu
(R2 ≈ 0.52 na random test split, nie jest to model do odkrywania leków).

Kontekst projektu:
{short_context}"""


def execute_predict_pic50(
    arguments: str,
    *,
    provider_label: str,
    llm_model: str,
    tool_round: int,
) -> dict[str, Any]:
    log_tool_event(
        "llm_tool_invoke",
        source="llm_assistant",
        provider=provider_label,
        llm_model=llm_model,
        tool="predict_pic50",
        tool_round=tool_round,
        arguments=arguments,
    )

    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as error:
        result = {"status": "error", "message": f"Invalid tool arguments: {error}"}
        log_tool_event(
            "llm_tool_result",
            source="llm_assistant",
            provider=provider_label,
            llm_model=llm_model,
            tool="predict_pic50",
            tool_round=tool_round,
            status="error",
            message=result["message"],
        )
        return result

    smiles = payload.get("smiles")
    if not isinstance(smiles, list) or not smiles:
        result = {"status": "error", "message": "Field 'smiles' must be a non-empty list of strings."}
        log_tool_event(
            "llm_tool_result",
            source="llm_assistant",
            provider=provider_label,
            llm_model=llm_model,
            tool="predict_pic50",
            tool_round=tool_round,
            status="error",
            message=result["message"],
        )
        return result

    smiles_list = [str(item).strip() for item in smiles if str(item).strip()]
    if not smiles_list:
        result = {"status": "error", "message": "No valid SMILES provided."}
        log_tool_event(
            "llm_tool_result",
            source="llm_assistant",
            provider=provider_label,
            llm_model=llm_model,
            tool="predict_pic50",
            tool_round=tool_round,
            status="error",
            message=result["message"],
        )
        return result

    try:
        result = predict_pic50_batch(
            smiles_list,
            source="llm_assistant",
            caller=f"{provider_label}:{llm_model}",
        )
    except WeightsNotFoundError as error:
        result = {"status": "error", "message": str(error)}
    except FileNotFoundError as error:
        result = {"status": "error", "message": str(error)}
    except Exception as error:  # noqa: BLE001 — tool errors must return JSON to the model
        result = {"status": "error", "message": str(error)}

    if "predictions" in result:
        ok_count = sum(1 for item in result["predictions"] if item.get("status") == "ok")
        log_tool_event(
            "llm_tool_result",
            source="llm_assistant",
            provider=provider_label,
            llm_model=llm_model,
            tool="predict_pic50",
            tool_round=tool_round,
            ok_count=ok_count,
            predictions=result["predictions"],
        )
    else:
        log_tool_event(
            "llm_tool_result",
            source="llm_assistant",
            provider=provider_label,
            llm_model=llm_model,
            tool="predict_pic50",
            tool_round=tool_round,
            status="error",
            message=result.get("message"),
        )
    return result


def run_tool_call(
    name: str,
    arguments: str,
    *,
    provider_label: str,
    llm_model: str,
    tool_round: int,
) -> str:
    if name == "predict_pic50":
        result = execute_predict_pic50(
            arguments,
            provider_label=provider_label,
            llm_model=llm_model,
            tool_round=tool_round,
        )
    else:
        result = {"status": "error", "message": f"Unknown tool: {name}"}
        log_tool_event(
            "llm_tool_unknown",
            source="llm_assistant",
            provider=provider_label,
            llm_model=llm_model,
            tool=name,
            tool_round=tool_round,
        )
    return json.dumps(result, ensure_ascii=False)


def make_openai_client(api_key: str, base_url: str | None) -> OpenAI:
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def generate_tool_assistant_response(
    question: str,
    project_context: str,
    api_key: str,
    model: str,
    base_url: str | None = None,
    provider_label: str = "API",
) -> str:
    if not question.strip():
        return "Wpisz pytanie o projekt, modele, metryki albo predykcję pIC50 ze SMILES."
    if not api_key.strip():
        return f"Podaj klucz API ({provider_label}) w panelu bocznym."

    client = make_openai_client(api_key.strip(), base_url)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(project_context)},
        {"role": "user", "content": question.strip()},
    ]

    log_tool_event(
        "llm_request_start",
        source="llm_assistant",
        provider=provider_label,
        llm_model=model,
        question=question.strip()[:500],
    )

    try:
        tool_round = 0
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[PREDICT_PIC50_TOOL],
                tool_choice="auto",
                temperature=0.2,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                text = (message.content or "").strip()
                log_tool_event(
                    "llm_request_done",
                    source="llm_assistant",
                    provider=provider_label,
                    llm_model=model,
                    tool_used=tool_round > 0,
                    tool_rounds=tool_round,
                    response_preview=text[:300],
                )
                return text or "Model zwrócił pustą odpowiedź."

            tool_round += 1
            tool_names = [call.function.name for call in message.tool_calls]
            log_tool_event(
                "llm_tool_calls",
                source="llm_assistant",
                provider=provider_label,
                llm_model=model,
                tool_round=tool_round,
                tools=tool_names,
            )

            messages.append(message.model_dump(exclude_none=True))
            for tool_call in message.tool_calls:
                tool_result = run_tool_call(
                    tool_call.function.name,
                    tool_call.function.arguments,
                    provider_label=provider_label,
                    llm_model=model,
                    tool_round=tool_round,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

        log_tool_event(
            "llm_request_done",
            source="llm_assistant",
            provider=provider_label,
            llm_model=model,
            tool_used=True,
            tool_rounds=tool_round,
            status="tool_round_limit",
        )
        return "Osiągnięto limit wywołań narzędzi. Spróbuj uprościć pytanie."
    except APIConnectionError:
        if base_url and "11434" in base_url:
            return (
                "Nie udało się połączyć z Ollama. Uruchom Ollama i pobierz model z obsługą narzędzi, np.:\n\n"
                "```powershell\nollama pull llama3.1\nollama serve\n```"
            )
        return f"Nie udało się połączyć z {provider_label}. Sprawdź połączenie sieciowe."
    except APIStatusError as error:
        if error.status_code == 429:
            return (
                f"{provider_label} zwróciło błąd 429 (limit/kwota). "
                "Spróbuj OpenCode Zen z darmowym modelem (np. deepseek-v4-flash-free) "
                "albo lokalnej Ollama z narzędziami."
            )
        return f"{provider_label} zwróciło błąd: {error.message}"
    except Exception as error:  # noqa: BLE001 — show safe message in UI
        return f"Nie udało się uzyskać odpowiedzi z {provider_label}: {error}"


def generate_provider_response(
    question: str,
    project_context: str,
    provider_key: str,
    api_key: str,
    model: str,
) -> str:
    provider = TOOL_PROVIDERS[provider_key]
    resolved_key = api_key.strip() or (provider.fixed_api_key or "")
    return generate_tool_assistant_response(
        question=question,
        project_context=project_context,
        api_key=resolved_key,
        model=model,
        base_url=provider.base_url,
        provider_label=provider.label,
    )


def generate_openai_response(
    question: str,
    project_context: str,
    api_key: str,
    model: str = DEFAULT_OPENAI_MODEL,
) -> str:
    return generate_provider_response(question, project_context, "openai", api_key, model)
