# Food-image-retrieval-system
The design of features and methods to improve performance of an image retrieval system for 10 types of Malaysian food.

Scripts were written by me and group partner:
1. computeDistances.py
2. computeFeatures.py
3. getCodebook.py

Scripts provided for assignment:
1. featureExtraction.py
2. fullEval.py
3. queryEval.py


On anaconda prompt:
1. To run full evaluation of database:-
	> ```python fullEval.py```
2. To retrieve a different number of images
	> ```python fullEval.py -r 150```
2. To run a retrieval based on image '100.jpg' from fooddb
	> ```python queryEval.py -d 100```
