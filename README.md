# Ball-recognition-model
  
## Cel projektu
Celem projektu jest stworzenie modelu do detekcji położenia piłki na obrazie, a więc taki który zwraca współrzędne futbolówki. 

## Słowo wstępne
Przyjętym założeniem jest, że piłka znajduje się na obrazku.
Projekt jest więc łatwo rozszerzalny -> można np. utworzyć drugi model klasyfikacyjny, a więc zwracający informację czy na danym obrazie znajduje się piłka, po czym oba te modele złączyć w jeden program. Może być to wykorzystywane przy odpowiednim wycięciu zdjęcia np. do zweryfikowania pozycji spalonej.

Model jest stworzony w oparciu o [keras](https://keras.io/), a więc API pozwalające na łatwe tworzenie customowych modelów.<br/>
Użyte środowisko to [Google Colab](https://research.google.com/colaboratory/), który pozwala na wykonywanie kody na maszynach udostępnionych przez Google'a.<br/>
Pod koniec wykonywania projektu zdecydowalismy się wybrać Google Colab w poszserzonej wersji, aby łatwiej było dostać pamięć RAM.<br/>
Do repo dodane są też skrypty, które napisaliśmy, a bardzo ułatwiały pracę:<br/>
  - png_to_jpg.py - prosty skrypt konwertujący wszyskie obrazy w wybranym folderze.
  - yt_cut.py - skrypt, który z filmiku na wejściu wycina co N-tą klatkę, dając dużą ilość zdjęć.</br>
Ostateczny model składa się z warstw jak poniżej, przetrenowany został na datasecie złożonym z ~53000 obrazów:<br/>
![Screenshot](zdj/final_model.jpeg)

Ale wszystko po kolei:

## Zebranie danych

Zdecydowaliśmy się na dane wysokiej jakości, aby piłka była dobrze widoczna. Na początku gromadziliśmy wszystkie dostępne zdjęcia, jednak potem zasadne było ograniczenie się do takich zdjęcia na których piłka jest wyraźna. Staraliśmy się tez wybierać z mała piłką jednak taka klasyfikacja nie ma sensu ze wzglądu na późniejsze przetwarzanie uniemożliwia znalezienie tylko kilku pikseli gdzie znajduje się futbolówka. Wybraliśmy piłki różnego rodzaju, z różnych rozgrywek m.in Premier League, mistrzostwa świata 2018, Liga Mistrzów. Nasz proces polegał na selekcji jakościowych momentów wycięcie ich i wycinanie odpowiednich klatek. Następnie zaznaczenie piłek tam gdzie występują. Korzystaliśmy z programu na którego wejście podawaliśmy co którą klatkę wycinamy i z których części pliku.

## Przygotowanie danych

Początkowo zgromadzono ~200 zdjęć oraz oznaczono piłke na każdym z nich, przy użyciu narzędzia [labelimg](https://github.com/tzutalin/labelImg).
Utworzono także pierwszy podstawowy model, jednak przewidywał on obiekt zawsze na środku obrazka nie ważne czy była to trawa czy piłkarz.
Przykłady:<br/>
![Screenshot](zdj/bad_predictions.jpeg)

Po wynikach loss i validation loss mozna zobaczyć, że model się przeuczał:<br/>
![Screenshot](zdj/overfitted.jpeg)

Najlepiej jednak dane te przedstawić w formie wykresu:<br/>
![Screenshot](zdj/overfitted_chart.jpeg)

Skąd wiadomo, że model jest przuczony?
W zasadzie wystarczy zobaczyć, że treningowy loss wciąż spada, a ten walidacyjny przestaje się już poprawiać.
Przykładowe wytłumaczenie na wiki:
[overfitting_wiki](https://en.wikipedia.org/wiki/Overfitting#Machine_learning)

Standardowo przy przeuczaniu trzeba zdobyc więcej danych treningowych, pomaga w tym augmentacja.<br/>
(cyt. *Augmentacja danych polega na wprowadzeniu do materiału treningowego nieco zmodyfikowanych kopii istniejących danych, co zazwyczaj przekłada się pozytywnie na wyniki algorytmów uczenia maszynowego.*)

- W naszym przypadku augmentacja polegała na:
  - ucinaniu rogów obrazka -> daje to dwa nowe z jednego oraz powoduje, że piłki są bliżej krawędzi obrazków.
  - flipowaniu obrazków
  - zmianach w kontraście obrazków
  - zmianach w jasności obrazków

Z oryginalnego obrazka (FHD) powstały cztery nowe z przycięć (3/4 FHD) oraz flipów, a z nich 14 ze zmian w kontraście oraz jasności.
Model na wejście przyjmuje 270x480px (1/4FHD), do tych wymiarów orazki musiały byc skalowane, więc przycinanie rogów nie powodowało żadnych problemów.
Dataset treningowy powstał z dodania do siebie:
- oryginał + cropped + flipped + contrast + brightness
co daje:
- 1 + 1*4 + 4*14 = 429
Zebrano ~1300 zdjęć, a więc 61*1300=79300, jednak po cropie piłka nie zawsze znajdowała się na obrazku, więc niektóre trzeba było odrzucić.
Ostatecznie dataset miał w sobie ~53k zdjęć.

Problemem w tym momencie była pamięć RAM, bo wczytanie takiej ilości danych wymaga dużej ilości zasobów.
Jak dużej?
Zdjęcie jest wymiarów 270x480px, w formacie RGB, a więc każdy piksel przechowuje informację o tych trzech kolorach.
Powstały numpy.array ma więc wymiary ~53000x270x480x3, a każda z tych komórek jest wypełniona najmniejszym mozliwym floatem -> float16.
Najmniejszym, bo większy nie jest potrzebny.
Floatem, bo RGB przyjmuje wartości 0-255, ale potrzebna była normalizacja (o normalizacji jest info niżej)
Wystarczy więc pomnożyć wszystkie wymiary oraz 16 i wychodzi liczba bitów potrzebnej pamięci.
53000*270*480*3*16 bitów, co daje ~38Gb.

Trzeba więc, aby model czytał dane w locie i podawał do modelu, można to osiągnąć poprzez stworzenie dataseta oraz funkcji służącej jako wejściowy pipeline.<br/>
[Użyta biblioteka do dataseta](https://www.tensorflow.org/datasets/api_docs/python/tfds)<br/>
Po zrobieniu tego wyniki były już lepsze, jednak kilka rzeczy jeszcze zostało do zrobienia:
  - Normalizacja danych
  - Znalezienie najlepszego modelu<br/><br/>
Normalizacja została wykonana w zasadzie w trakcie robienia datasetu, polegała na tym, żeby zamienić zapis zdjęć z formy klasycznego RGB, a więc (0-255) na (0-1), jest to zrobione przy wejściowym pipelinie - tablica jest dzielona przez 255, a wykorzystywany typ danych to Float16.
Tak samo dla BBox'ów(Bouding Box - kwadraty wyznaczające położenie piłki, składające się z ((Xmin, Ymin), (Xmax, Ymax)), a więc dwóch punktów) wystarczyło przedzielic przez 480 -> najwiekszy wymiar obrazu.

Do szukania modelu użyto [autotuner'a](https://www.tensorflow.org/tutorials/keras/keras_tuner), który pozwala dobrać odpowiednie wawrtości dla w zasadzie wszystkiego, a więc liczby filtrów, liczby jednostek, szybkości uczenia, liczby warstw i wiele innych.
Przykładowe użycie:<br/>
![Screenshot](zdj/autotuner.jpeg)

### Przykłady augmentacji

  - Obrazek oryginalny<br/>
  ![Screenshot](zdj/original.jpeg)
  - Ucięty dolny prawy róg (Crop1)<br/>
  ![Screenshot](zdj/crop1.jpeg)
  - Ucięty górny lewy róg (Crop2)<br/>
  ![Screenshot](zdj/crop2.jpeg)
  - Crop1 flipped<br/>
  ![Screenshot](zdj/crop1_flipped.jpeg)
  - Crop2 flipped<br/>
  ![Screenshot](zdj/crop2_flipped.jpeg)

## Model

- Model jest bazowany na warstwach splotowych, a więc składa się z warstw:
  - Conv2D -> wprowadzającej splot
  - MaxPooling2D -> zmieniającej rodzielczość obrazka
  - Dropout -> przepuszczającej tylko fragment danych (walka z przeuczeniem)
  - Dense -> w pełni połączone neurony
- ADAM optymalizator -> jest on najbardziej uniwersalny
- Autotuner -> narzędzie które pomaga ustalić odpowiednie wartości ilości filtrów w warstwach splotowych, unitów w warstwach dense oraz learning rate
- Rysowanie krzywej uczenia, (zmiany funkcji loss dla datasetu treningowego oraz walidacyjnego) pozwala to określić czy model się przeucza.

## Wyniki

Funkcja loss dobrego modelu:<br/>
![Screenshot](zdj/loss_function.jpeg)

Przykłady zdjęć z przewidywaniami:<br/>
![Screenshot](zdj/examples.png)

## Przykłady działania, na 100 zdjęciach ze zbioru walidacyjnego
<img src="zdj/examples/1.jpeg" width="18%"></img> 
<img src="zdj/examples/2.jpeg" width="18%"></img> 
<img src="zdj/examples/3.jpeg" width="18%"></img> 
<img src="zdj/examples/4.jpeg" width="18%"></img> 
<img src="zdj/examples/5.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/6.jpeg" width="18%"></img> 
<img src="zdj/examples/7.jpeg" width="18%"></img> 
<img src="zdj/examples/8.jpeg" width="18%"></img> 
<img src="zdj/examples/9.jpeg" width="18%"></img> 
<img src="zdj/examples/10.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/11.jpeg" width="18%"></img> 
<img src="zdj/examples/12.jpeg" width="18%"></img> 
<img src="zdj/examples/13.jpeg" width="18%"></img> 
<img src="zdj/examples/14.jpeg" width="18%"></img> 
<img src="zdj/examples/15.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/16.jpeg" width="18%"></img> 
<img src="zdj/examples/17.jpeg" width="18%"></img> 
<img src="zdj/examples/18.jpeg" width="18%"></img> 
<img src="zdj/examples/19.jpeg" width="18%"></img> 
<img src="zdj/examples/20.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/21.jpeg" width="18%"></img> 
<img src="zdj/examples/22.jpeg" width="18%"></img> 
<img src="zdj/examples/23.jpeg" width="18%"></img> 
<img src="zdj/examples/24.jpeg" width="18%"></img> 
<img src="zdj/examples/25.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/26.jpeg" width="18%"></img> 
<img src="zdj/examples/27.jpeg" width="18%"></img> 
<img src="zdj/examples/28.jpeg" width="18%"></img> 
<img src="zdj/examples/29.jpeg" width="18%"></img> 
<img src="zdj/examples/30.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/41.jpeg" width="18%"></img> 
<img src="zdj/examples/42.jpeg" width="18%"></img> 
<img src="zdj/examples/43.jpeg" width="18%"></img> 
<img src="zdj/examples/44.jpeg" width="18%"></img> 
<img src="zdj/examples/45.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/46.jpeg" width="18%"></img> 
<img src="zdj/examples/47.jpeg" width="18%"></img> 
<img src="zdj/examples/48.jpeg" width="18%"></img> 
<img src="zdj/examples/49.jpeg" width="18%"></img> 
<img src="zdj/examples/50.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/51.jpeg" width="18%"></img> 
<img src="zdj/examples/52.jpeg" width="18%"></img> 
<img src="zdj/examples/53.jpeg" width="18%"></img> 
<img src="zdj/examples/54.jpeg" width="18%"></img> 
<img src="zdj/examples/55.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/56.jpeg" width="18%"></img> 
<img src="zdj/examples/57.jpeg" width="18%"></img> 
<img src="zdj/examples/58.jpeg" width="18%"></img> 
<img src="zdj/examples/59.jpeg" width="18%"></img> 
<img src="zdj/examples/60.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/61.jpeg" width="18%"></img> 
<img src="zdj/examples/62.jpeg" width="18%"></img> 
<img src="zdj/examples/63.jpeg" width="18%"></img> 
<img src="zdj/examples/64.jpeg" width="18%"></img> 
<img src="zdj/examples/65.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/66.jpeg" width="18%"></img> 
<img src="zdj/examples/67.jpeg" width="18%"></img> 
<img src="zdj/examples/68.jpeg" width="18%"></img> 
<img src="zdj/examples/69.jpeg" width="18%"></img> 
<img src="zdj/examples/70.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/71.jpeg" width="18%"></img> 
<img src="zdj/examples/72.jpeg" width="18%"></img> 
<img src="zdj/examples/73.jpeg" width="18%"></img> 
<img src="zdj/examples/74.jpeg" width="18%"></img> 
<img src="zdj/examples/75.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/76.jpeg" width="18%"></img> 
<img src="zdj/examples/77.jpeg" width="18%"></img> 
<img src="zdj/examples/78.jpeg" width="18%"></img> 
<img src="zdj/examples/79.jpeg" width="18%"></img> 
<img src="zdj/examples/80.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/81.jpeg" width="18%"></img> 
<img src="zdj/examples/82.jpeg" width="18%"></img> 
<img src="zdj/examples/83.jpeg" width="18%"></img> 
<img src="zdj/examples/84.jpeg" width="18%"></img> 
<img src="zdj/examples/85.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/86.jpeg" width="18%"></img> 
<img src="zdj/examples/87.jpeg" width="18%"></img> 
<img src="zdj/examples/88.jpeg" width="18%"></img> 
<img src="zdj/examples/89.jpeg" width="18%"></img> 
<img src="zdj/examples/90.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/91.jpeg" width="18%"></img> 
<img src="zdj/examples/92.jpeg" width="18%"></img> 
<img src="zdj/examples/93.jpeg" width="18%"></img> 
<img src="zdj/examples/94.jpeg" width="18%"></img> 
<img src="zdj/examples/95.jpeg" width="18%"></img> 
<br/> 
<img src="zdj/examples/96.jpeg" width="18%"></img> 
<img src="zdj/examples/97.jpeg" width="18%"></img> 
<img src="zdj/examples/98.jpeg" width="18%"></img> 
<img src="zdj/examples/99.jpeg" width="18%"></img> 
<img src="zdj/examples/100.jpeg" width="18%"></img> 
<br/> 


## Opiekun Merytoryczny
<table align="center">
  <tr align="center">
    <td>Jarosław 'Dzik Merytoryczny' Bułat</td>
  </tr>
  <tr align="center">
    <td><img src="zdj/JB.jpeg" width=50%></td>
  </tr>
</table>


## Autorzy

<table>
  <tr align="center">
    <td>Kamil 'Mufasa' Sobolak</td>
    <td>Kacper 'Whitecore' Zemła</td>
    <td>Marek 'Legio' Kwak</td>
  </tr>
  <tr align="center">
    <td><img src="zdj/KS.jpg" width=100%></td>
    <td><img src="zdj/KZ.jpeg" width=100%></td>
    <td><img src="zdj/MK.jpeg" width=100%></td>
  </tr>
</table>
