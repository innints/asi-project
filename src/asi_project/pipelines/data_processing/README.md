# pipelines/data_processing

Potok czyta ramkę danych `raw_data` i zwraca `prepared_data` z etykietą targetu,
gotowe do podziału na zbiory treningowy i testowy.

1. `validate_data_contract` ujednolica nazwy kolumn i sprawdza dane za pomocą `assert_valid`,
   korzystając z konfiguracji projektu i rejestru datasetów.
2. `normalize_input` normalizuje wartości wejściowe.
3. `filter_training_rows` usuwa rekordy bez etykiety targetu.

Wszystkie trzy kroki korzystają z funkcji z `src/asi_project/data.py`.
Kedro wykrywa potok przez `create_pipeline()`. Można to sprawdzić poleceniem
`kedro registry list`. Data Catalog określa źródło `raw_data` i zapis `prepared_data`
w `data/02_intermediate/`.
