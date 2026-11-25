# Porównanie modeli — Sprint 3

Wszystkie modele oceniono na tym samym zbiorze testowym utworzonym przy użyciu parametrów `test_size: 0.2` i `random_state: 42` z konfiguracji projektu.

| Model | F1 | Czas predykcji zbioru testowego [ms] | Czas na rekord [ms] | Rozmiar artefaktu [MB] | RAM po wczytaniu [MB] |
|---|---:|---:|---:|---:|---:|
| baseline-logistic-regression | 0.5442 | 1.97 | 0.0031 | 0.004 | 71.91 |
| autogluon-medium-60s | 0.8932 | 4.52 | 0.0072 | 35.134 | 93.29 |
| autogluon-good-120s | 0.9010 | 6.93 | 0.0110 | 64.422 | 92.81 |

## Metoda pomiaru

- zbiór testowy: 630 rekordów;
- czas predykcji: mediana z 5 powtórzeń po jednej rozgrzewce;
- rozmiar artefaktu: suma rozmiarów wszystkich plików modelu na dysku;
- RAM po wczytaniu: przyrost pamięci RSS po załadowaniu modelu w osobnym, czystym procesie;
- główna metryka projektu: F1 dla klasy pozytywnej `Churn=1`.

## Wniosek

Najwyższe F1 osiągnął `autogluon-good-120s`: 0.9010. Ostateczny wybór modelu produkcyjnego powinien uwzględniać również koszt pamięci, rozmiar artefaktu i czas predykcji przedstawione w tabeli.
