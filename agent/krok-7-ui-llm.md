# Krok 7: prosty UI z lokalnym asystentem tekstowym i Ollama

## Cel

Celem tego kroku było dodanie prostego interfejsu użytkownika oraz lekkiej funkcjonalności typu LLM. Asystent odpowiada wyłącznie tekstem i nie wykonuje żadnych akcji poza wygenerowaniem odpowiedzi.

## Wykonane zmiany

Dodano zależność:

```text
streamlit
```

Tryb Ollama korzysta z biblioteki `requests`, która była już używana w projekcie.

Dodano główny plik aplikacji:

```text
app.py
```

Aplikacja pokazuje:

- krótki opis projektu,
- najlepszy wynik modelu,
- porównanie najlepszego GCN i MLP,
- ograniczenia wyniku,
- pełne podsumowanie pracy z `agent/podsumowanie-pracy.md`,
- pole tekstowe do zadania pytania o projekt.

## Jak działa lokalny asystent

Aplikacja ma dwa tryby odpowiedzi wybierane w panelu bocznym:

- `Ollama lokalnie` - próbuje odpytać lokalny model przez `http://localhost:11434/api/generate`,
- `Lokalny stub` - używa prostych reguł tekstowych bez prawdziwego modelu LLM.

Domyślny model dla Ollama:

```text
llama3.2:3b
```

To mały model dobry na start, bo powinien być relatywnie lekki dla lokalnego uruchamiania.

## Tryb Ollama

Tryb Ollama jest zaimplementowany jako funkcja:

```text
generate_ollama_response(question, project_context, model)
```

Funkcja buduje prompt z krótkim kontekstem projektu i wysyła go do lokalnej Ollama. Prompt jawnie ogranicza model:

- odpowiedź ma być tylko tekstem,
- model nie ma wywoływać narzędzi,
- model nie ma uruchamiać kodu,
- model nie ma pobierać danych,
- model nie ma trenować modeli,
- model nie ma modyfikować plików.

Jeżeli lokalny serwer Ollama nie działa albo model nie jest pobrany, aplikacja pokazuje instrukcję naprawy zamiast przerywać działanie.

Podczas implementacji komenda `ollama --version` nie była dostępna w systemie, więc sama integracja została dodana, ale do pełnego użycia trzeba jeszcze zainstalować Ollama i pobrać model.

## Tryb stub

Asystent jest zaimplementowany jako lokalna funkcja:

```text
generate_text_response(question, project_context)
```

Funkcja działa deterministycznie na podstawie prostych reguł i lokalnego kontekstu projektu. Jest zostawiona jako fallback, gdy Ollama nie jest jeszcze zainstalowana albo uruchomiona.

Asystent może odpowiadać na pytania m.in. o:

- `pIC50` i `IC50`,
- dataset EGFR z ChEMBL,
- najlepszy wynik modelu,
- baseline MLP,
- model GCN,
- `random split` i `scaffold split`,
- ograniczenia ewaluacji.

Jeżeli pytanie wykracza poza prosty kontekst aplikacji, UI zwraca tekstową informację, że nie ma wystarczającego kontekstu.

## Ograniczenia

Asystent:

- nie wywołuje narzędzi,
- nie uruchamia treningu modeli,
- nie modyfikuje plików,
- nie korzysta z kluczy API,
- nie wykonuje poleceń systemowych.

Tryb Ollama wysyła prompt tylko do lokalnego serwera Ollama, a nie do zewnętrznego API. To jest prosty interfejs demonstracyjny, a nie pełny agent.

## Jak przygotować Ollama

Po instalacji Ollama można pobrać mały model:

```powershell
ollama pull llama3.2:3b
```

Jeśli Ollama nie działa jako aplikacja w tle, można uruchomić serwer:

```powershell
ollama serve
```

Aplikacja domyślnie odpytuje:

```text
http://localhost:11434/api/generate
```

## Jak uruchomić

Po aktywacji środowiska:

```powershell
.\.venv\Scripts\Activate.ps1
```

aplikację można uruchomić poleceniem:

```powershell
streamlit run app.py
```

Alternatywnie, bez aktywacji środowiska:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## Wynik kroku

Projekt ma teraz prosty UI, który pozwala szybko pokazać wyniki pracy i zadać tekstowe pytanie o projekt. Może działać w trybie lokalnego stubu albo z prawdziwym lokalnym LLM przez Ollama.
