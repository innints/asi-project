# conf/ — konfiguracja Kedro

> Sprint 2 — konfiguracja startowa Kedro. Nody i pipeline’y implementuje zespół.

Aktualizacja referencyjna na start Sprintu 2 przygotowała sekcję `[tool.kedro]`
w `pyproject.toml` i strukturę konfiguracji w tym katalogu.

| Katalog | Sprint | Rola |
|---|---|---|
| `conf/base/` | S2 (rozszerzany w S3) | konfiguracja wspólna: katalog danych i parametry |
| `conf/local/` | S2 | ustawienia lokalne i poświadczenia — poza gitem (oprócz README) |
| `conf/ci/` | S2 | nadpisania dla CI: te same pipeline'y na próbce `data/01_raw/sample.csv` |

Kontrakt (z planu Sprintu 2):

- `kedro run` — pełne dane (`base` + `local`);
- `kedro run --env ci` — próbka CI (`base` + `ci`), uruchamiane w GitHub Actions;
- kod pipeline'ów nie zna ścieżek do plików — ścieżki i parametry są w konfiguracji.

Wpisy w `conf/base/catalog.yml` dla Iranian Churn:

| Nazwa w Kedro | Rola | Plik |
|---|---|---|
| `raw_data` | Wejście `data_processing` | `data/01_raw/iranian_churn/Customer Churn.csv` |
| `prepared_data` | Wyjście `data_processing`, wejście `data_science` | `data/02_intermediate/prepared_data.csv` |
| `baseline_model` | Wytrenowany sklearn Pipeline | `data/06_models/baseline_model.pkl` |
| `metrics` | Wyjście węzła ewaluacji | `data/08_reporting/metrics.json` |

Pełny zbiór pobiera `python scripts/download_data.py`. Dane i artefakty wymienione
w tabeli są przechowywane lokalnie poza Gitem.

Przypięte wersje Kedro i `kedro-datasets`: `requirements/preflight-s2.txt`.

**Najpierw sprawdź inicjalizację** w katalogu głównym projektu:

```bash
python -m pip install -r requirements/preflight-s2.txt -c constraints.txt
python -m pip install --no-deps -e .
kedro registry list
```

Jeśli Kedro rozpoznaje projekt, polecenie `kedro registry list` wykonuje się
bez błędu "Kedro project not found". Na tym etapie registry może zawierać
jedynie pusty `__default__` — to oczekiwane, nie ma jeszcze nodów.

Na tym etapie mogą pojawić się ostrzeżenia o braku `create_pipeline()` w modułach `data_processing` i `data_science`. Są oczekiwane i znikną, gdy zespół zaimplementuje te funkcje.

**Następnie zadaniem zespołu** jest utworzenie `create_pipeline()` w
`data_processing` i `data_science`, skonfigurowanie parametrów oraz
uruchomienie pełnego procesu: `kedro run` i `kedro run --env ci`.

Nie uruchamiaj `kedro new` wewnątrz istniejącego repozytorium. Ponieważ
repozytoria zespołowe powstały wcześniej z template, zmiana w `asi-reference`
nie trafia do nich automatycznie: trzeba przenieść aktualizację referencyjną.
