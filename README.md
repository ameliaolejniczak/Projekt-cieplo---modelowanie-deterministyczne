# Modelowanie rozkładu ciepła
Repozytorium zawiera projekt modelujący ogrzewanie dla dwóch problemów. W pierwszym z nich rozważamy znaczenie położenia grzejnika w stosunku do okna w ogrzewanym pomieszczeniu. W drugim z nich spojrzymy na ogrzewanie sąsiednich pokoi i jakie znaczenie ma ogrzewanie wszystkich pokoi. 

## Zawartość
- folder ```\pipelines``` z plikami zawierającymi klasy potrzebne do działania kodu, a także animacje ze zmianą ciepła dla obu problemów oraz kod liczący błąd podwojonego kroku dla $h_t$,
- folder ```\data``` z tabelą z danymi teoretycznymi w formacie .csv
- forder ```\notebooks``` z notatnikami Jupyter z krótkim opisem i wnioskami rozważanych problemów.

## Instalacja
1. Sklonuj repozytorium
   ```
   git clone https://github.com/ameliaolejniczak/Projekt-cieplo---modelowanie-deterministyczne.git
   ```
2. Zainstaluj wymagane biblioteki
   ```
   pip install -r requirements.txt
   ```

## Uruchomienie kodu
Przy pomocy VS Code lub Jupyter Notebbok można uruchomić notatniki z folderu ```\notebooks```. Można również uruchomić skrypty ```animacja_problem1.py``` oraz ```animacja_problem2.py``` aby zobaczyć wizualizacje zmiany temperatury w obu problemach.



