# Wybór modelu produkcyjnego i formatu pakietu modelu

## Kontekst i problem

W sprincie 3 porównaliśmy model bazowy ze sprintu 2 (`baseline-logistic-regression`) z kilkoma konfiguracjami AutoGluon (`medium_quality` i `good_quality` dla limitów czasu 60s i 120s) na zbiorze `iranian_churn` (metryka główna: F1-score, podział `stratified_random` 80/20, `random_state=42`).

Musieliśmy wybrać jeden model, który trafi jako artefakt `production` do W&B i będzie można go łatwo załadować w API bez potzreby ponownego trenowania.

## Rozważane opcje

| Model | F1 (test) | Czas predykcji (630 wierszy) | Rozmiar paczki | RAM po wczytaniu |
|---|---|---|---|---|
| `baseline-logistic-regression` | `0.5442` | `1.97 ms` | `0.004 MB` | `71.91 MB` |
| `autogluon-medium-60s` | `0.8932` | `4.52 ms` | `35.134 MB` | `93.29 MB` |
| `autogluon-good-120s` | `0.9010` | `6.93 ms` | `64.422 MB` | `92.81 MB` |

## Decyzja

1. **Wybór modelu produkcyjnego: `autogluon-medium-60s`** (najlepszy podmodel: `WeightedEnsemble_L2` oparty na `LightGBM`, opublikowany w W&B jako `iranian-churn-model:production`).
2. **Format pakietu modelu (`src/asi_project/model_package.py`):**
   - Katalog pakietu zawiera natywny katalog predyktora AutoGluon (`predictor/`) oraz plik `metadata.json`.
   - `metadata.json` przechowuje: `model_name`, `model_format`, `best_submodel`, listę `features`, `target`, `positive_class`, słownik `metrics`, `git_sha`, `data_fingerprint` oraz `library_versions`.
   - Cały katalog publikowany jest jako jeden `wandb.Artifact` typu `model` z aliasem `production`.

## Uzasadnienie

- **Dlaczego nie `baseline-logistic-regression`?** Mimo tego, że rozmiar oraz czas predykcji jest minimalny, jakość predykcji (`F1 = 0.5442`) jest za niska dla zastosowań CRM, ponieważ za dużo jest niewykrytych odejść klientów.
- **Dlaczego nie `autogluon-good-120s`?** Zysk jakościowy od `autogluon-medium-60s` różni się niewiele, bo tylko 0.0078, ale koszt zasobów rośnie prawie dwukrotnie. Zurzycie RAM rośnie o około 31 MB, a czas predylkacji wydłuża się aż o 50%.
- **Dlaczego `autogluon-medium-60s` zamiast `120s`?** Wynik jakościowy i rozmiar jest identyczny jak w wariancie 120-sekundowy przy krótszym czasie treningu.
- **Format paczki (`predictor/` + `metadata.json`):** Oddziela ładowanie modelu od konkretnej biblioteki (wspólne funkcje `save_model_package` / `load_model_package` obsługują zarówno `joblib`, jak i `autogluon`) i gwarantuje pełną odtwarzalność w API.

## Co z tego wynika

- Osoba budująca API nie musi nic trenować — po prostu pobiera gotowy artefakt `s29848-pjatk/asi-project/iranian-churn-model:production` z W&B i pilnuje, żeby kolejność kolumn zgadzała się z `features` z `metadata.json`.
- W środowisku z API trzeba mieć zainstalowaną tę samą wersję `autogluon.tabular==1.6.3`, żeby paczka z modelem wczytała się bez błędów.