# pipelines/data_science

> Scaffold for Sprint 2 — do not implement before this sprint. Rozszerzany w Sprincie 3.

- Sprint 2: split zgodny z kontraktem datasetu → trening sklearn Pipeline (model preprocessing
  i model) → ewaluacja. Wyjścia: model (`data/06_models/`) i metryki (`data/08_reporting/`).
- Sprint 3: runy W&B (parametry, metryki, git SHA, fingerprint danych), node AutoGluona
  z limitem czasu, świadomy wybór modelu i pakiet modelu z aliasem `production`.

`training.create_training_nodes()` udostępnia węzły podziału i treningu do połączenia
z ewaluacją w `create_pipeline()`. Ich wejściem jest `prepared_data`, a wyjścia to
`X_train`, `X_test`, `y_train`, `y_test` i `baseline_model`.

- Podział pobiera `params:split.test_size` i `params:split.random_state`.
- Trening pobiera `params:logistic_regression`, w tym `random_state` i `max_iter`.
- Przy zapisie metryk użyj tych samych parametrów podziału z Kedro, żeby opis metryk
  uwzględniał też wartości nadpisane dla danego uruchomienia.
