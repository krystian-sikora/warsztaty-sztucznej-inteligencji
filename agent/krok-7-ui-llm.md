# Krok 7: prosty UI z lokalnym asystentem tekstowym

## Cel

Celem tego kroku było dodanie prostego interfejsu użytkownika oraz lekkiej funkcjonalności typu LLM. W tej wersji asystent odpowiada wyłącznie tekstem i nie wykonuje żadnych akcji poza wygenerowaniem odpowiedzi.

## Wykonane zmiany

Dodano zależność:

```text
streamlit
```

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

Asystent jest zaimplementowany jako lokalna funkcja:

```text
generate_text_response(question, project_context)
```

Nie jest to połączenie z zewnętrznym API. Funkcja działa deterministycznie na podstawie prostych reguł i lokalnego kontekstu projektu.

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
- nie pobiera danych z internetu,
- nie korzysta z kluczy API,
- nie wykonuje poleceń systemowych.

To jest prosty interfejs demonstracyjny, a nie pełny agent.

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

Projekt ma teraz prosty UI, który pozwala szybko pokazać wyniki pracy i zadać tekstowe pytanie o projekt bez używania zewnętrznego LLM.
