# Model Card

> Scaffold for Sprint 3 — do not implement before this sprint.
> Minimalną Model Card wypełnia zespół w Sprincie 3, a aktualizuje w Sprincie 6.

## Model

- Typ modelu i jego wersja (alias `production` w W&B): AutoGluon `TabularPredictor` (`autogluon-medium-60s`, najlepszy podmodel `WeightedEnsemble_L2` oparty na `LightGBM`, preset `medium_quality`, `time_limit=60s`), opublikowany w W&B jako `iranian-churn-model:production`
- Data utworzenia i autor runu: 2026-11-25, Maria Chmielewska

## Przeznaczenie

- Problem (binary classification) i klasa pozytywna: Klasyfikacja binarna przewidująca odejście klienta operatora telekomunikacyjnego. Klasa pozytywna: `Churn = 1` (oznacza odejście klienta)
- Użytkownik systemu i decyzja, którą wspiera model: Dział utrzymania klienta, model wskazuje klientów, którzy mają wyższe ryzyku rezygnacji, żeby wysłać im nową ofertę.
- Zastosowania, do których model nie jest przeznaczony: Automatyczne blokowanie lub wypowiadanie umów bez weryfikacji pracownika, ocena zdolności kredytowej oraz predykcja dla klientów z  innych branż

## Dane

- Dataset (id z `docs/datasets.md`) i fingerprint danych: `iranian_churn` (`Customer Churn.csv`, 3150 rekordów), SHA-256: `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1`.
- Strategia splitu z kontraktu i liczebności zbiorów: `stratified_random` (`test_size: 0.2`, `random_state: 42`) — zbiór treningowy: 2520 rekordów, zbiór testowy: 630 rekordów

## Metryki

- Metryka główna na zbiorze testowym, w porównaniu z DummyClassifier:
- `autogluon-medium-60s` (model produkcyjny): F1 = 0.8932
  - `baseline-logistic-regression` (Sprint 2): F1 = 0.5442
  - `DummyClassifier` (strategia `most_frequent` / `stratified`): F1 = 0.0000 / ~0.16 (znacząca poprawa względem baseline'u i modelu losowego).
- Pozostałe metryki i macierz pomyłek:
- Accuracy: `0.9651` | Precision: `0.8598` | Recall: `0.9293`
  - Macierz pomyłek: `TN = 516`, `FP = 15`, `FN = 7`, `TP = 92`
  - Wydajność: czas predykcji zbioru testowego `4.52 ms` (`0.0072 ms` / rekord), rozmiar paczki na dysku `35.134 MB`, RAM po wczytaniu `93.29 MB`.

## Ograniczenia i ryzyka

- Ograniczenia danych (z `docs/datasets.md`): Zbiór jest dość mały (3150 wierszy), pochodzi z jednego konkretnego rynku i opisuje klienta tylko ogólnymi statystykami z ostatnich miesięcy. W danych zostawiliśmy powtarzające się wiersze, bo dwóch różnych klientów może mieć identyczne statystyki użycia telefonu
- Znane błędy i sytuacje, w których model zawodzi: Przy nagłych zmianach cenników lub promocji konkurencji oraz dla nowych klientów z zerową historią połączeń/SMS

## Wersje

- Git SHA treningu: `ddc3e5a0f792e42104d227c9a414d6d8a2f79890`
- Wersje bibliotek: `autogluon.tabular==1.6.3`, `scikit-learn==1.7.2`, `pandas==2.3.3`, `numpy==2.2.6`, `joblib==1.6.0`, `wandb==0.30.0`.
