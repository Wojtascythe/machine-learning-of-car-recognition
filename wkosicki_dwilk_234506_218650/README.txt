Dostarczony pakiet uruchomieniowy składa się z trzech skryptów:

1) createTraningData.py - odpowiada za konwersję obrazów z kolorowych do szarych zmniejszając dodatkowo ich rozmiar do 64 x 64 pikseli -
podobnie jak zostało przedstawione w ćwiczeniu wzorcowym przedstawionym na stronie http://www.cse.chalmers.se/~richajo/dit866/PA3.html. 
Oryginalne obrazy są pobierane z czterech folderów: 'TrainImages/train/car/', 'TrainImages/train/other/', 'TrainImages/validation/car/', 'TrainImages/validation/other/'.
Obrazy po konwersji są zapisywane w: 'ResizeTrainImages/train/car/', 'ResizeTrainImages/train/other/', 'ResizeTrainImages/validation/car/'
'ResizeTrainImages/validation/other/'. 

2) carDetectionCNN.py - pobiera obrazy po konwersji z ww. folderów i tworzy model klasyfikatora wykorzystując
konwulsyjne sieci neuronowe. Klasyfikator jest zapisywany pod nazwą "carDetectionCNN.h5". 
Do pakietu uruchomieniowego dostarczono wcześniej wygenerowany klasyfikator.

3) Runme.py - dokonuje klasyfikacji obrazów znajdujących się w folderze, którego nazwa została przekazana jako argument wywołania skryptu:
python Runme.py NAZWA_FOLDERU np. python Runme.py ImagesToTest. Jeżeli nie zostanie przekazany żaden folder w argumencie wywołania,
wówczas dane domyślnie zostaną pobrane z folderu "test" znajdującego się w tej samej lokalizacji co Runme.py. 
Skyrpt wykorzystuje klasyfikator "carDetectionCNN.h5".
W trakcie działania skryptu wyświetlane są nazwy aktualnie przetwarzanego obrazu wraz z informacją czy zawiera samochód, czy nie.
Po zakończeniu wyświetlana jest ilość plików z rozpoznanymi pojazdami.






