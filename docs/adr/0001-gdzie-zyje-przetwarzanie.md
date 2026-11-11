# ADR-1: Gdzie żyją różne rodzaje przetwarzania?

- **Status:** Zatwierdzony
- **Sprint:** S2
- **Autor:** Urszula Kępka
- **Recenzent:** Maria Chmielewska

## Kontekst

Nasz system ma przewidywać klientów którzy będą odchodzić z firmy telekomunikacyjnej. Nasz kod w Pythonie przygotowywuje dane do treningu oraz trenuje model. W przyszłych sprintach musimy obsługiwać zapytania przychodzące przez API.

Musimy ustalić, gdzie w projekcie wykonywane są poszczególne przekształcenia: input normalization, training-only filtering i model preprocessing. Musimy uniknąć wycieku danych oraz sytuacji gdzie model na produkcji są nieco inaczej przygotowywane niż przy nauce.

## Rozważone opcje

1. Opcja A — Wszytsko w jednym miejscu: umieszczenie wszystkich kroków (czyszczenie danych, usuwanie wierszy, skalowanie cech i uzupełnianie braków średnią)
2. Opcja B — Podział na 3 części: input normalization, training-only filtering i model preprocessing

## Decyzja


Wybieramy opcję B

Podział:
- input normalization jest wspólne dla treningu i późniejszego API w związku z tym JSON przychodzący z API zostanie przetworzony w taki sam sposób co dane przy treningu. Żyje w pliku `data.py`. Wykorzystywane w potoku `data_processing`
- training-only filtering działa tylko podczas przygotowania danych treningowych. Znajduje się w pliku `data.py`, jest uruchamiane tylko w potoku treningowym Kedro nigdy przy predykcji API. Wykorzystywane w potoku   `data_processing`
- model preprocessing, czyli operacje wykorzystujące `.fit()` są wykonywane po podziale na zbiór treningowy i testowy, aby uniknąć wycieku danych. Żyje w `asi_project.model` wewnątrz `sklearn.Pipeline`. Wykorzystywane w potoku  `data_science`

| Wielkość | Wartość | Skąd pochodzi |
|---|---|---|
| Kolumny wykluczone | Customer Value | `docs/datasets.md` |
| Ilość rekordów | 3150 | `data/01_raw/Customer Churn.csv` `docs/datasets.md` |

## Odrzucone alternatywy

Odrzuciliśmy opcję A:
- wyciek danych, liczenie `.fit()` przed podziałem
- trzeba stworzyć nową funkcję dla API, bo ona byłaby zamknięta w jednym potoku

## Konsekwencje

Co ta decyzja ułatwia, co utrudnia i co trzeba będzie zrobić w kolejnych sprintach.

Ułatwia:
- Uniknięcie wycieku danych
- Te same funkcje input normalization dla treningu i API
- Powstanie jeden artefakt zawierający wszystko
Utrudnia:
- Pilnowanie się zespołu aby używać odpowiednio funkcji
- Przy tworzeniu API nalerzy pamiętać o input normalization

## Historia

| Data | Zmiana |
|---|---|
| 11.11.2026 | utworzenie |
