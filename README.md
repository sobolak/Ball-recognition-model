# Ball-recognition-model

Celem projektu jest stworzenie klasyfikatora obrazów opartego o sieć splotową. Przygotowaliśmy i wytrenowaliśmy model wykrywający piłki z meczów piłkarskich. Czy będzie to najnowszy model, czy zwykła biedronka na pewno sobie poradzimy.

#### Zebranie danych

Zdecydowaliśmy się na dane wysokiej jakości, aby piłka była dobrze widoczna. Na początku gromadziliśmy wszystkie dostępne zdjęcia, jednak potem zasadne było ograniczenie się do takich zdjęcia na których piłka jest wyraźna. Staraliśmy się tez wybierać z mała piłką jednak taka klasyfikacja nie ma sensu ze wzglądu na późniejsze przetwarzanie uniemożliwia znalezienie tylko kilku pikseli gdzie znajduje się futbolówka. Wybraliśmy piłki różnego rodzaju, z różnych rozgrywek m.in Premier League, mistrzostwa świata 2018, Liga Mistrzów. Nasz proces polegał na selekcji jakościowych momentów wycięcie ich i wycinanie odpowiednich klatek. Następnie zaznaczenie piłek tam gdzie występują.

#### Przygotowanie danych

Początkowo (po pierwszych wyciętych zdjęciach) stworzony model wyszukiwał obiekt zawsze na środku obrazka nie ważne czy była to trawa czy piłkarz. W dodatku wybór zawsze miał ten sam obraz. Udało nam się z tym poradzić gromadząc większą ilość danych wejściowych  oraz augmentacją. 

- Zmniejszenie przeuczenia
  - obcinanie rogów aby obiekt nie znajdował się w środku
  - obracanie wzdłuż krawędzi
  - dodanie szumu
  - jasność obrazka
  - Batch Normalization

#### Model

- autotuner
- znajdowanie tylko na środku

- z jakich warstw

#### Wyniki

- pokazanie jak wyszukuje 
- wykresiki