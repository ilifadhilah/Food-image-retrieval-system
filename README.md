# Food-image-retrieval-system
Design of features and methods to improve performance of image retrieval system for 10 types of Malaysian food

Scripts were written by me and group partner:
-computeDistances.py
-computeFeatures.py
-getCodebook.py

Scripts provided for assignment:
-featureExtraction.py
-fullEval.py
-queryEval.py

On anaconda prompt:
1. To run full evaluation of database:-
	>python fullEval.py
2. To retrieve a different number of images
	>python fullEval.py -r 150
2. To run a retrieval based on image '100.jpg' from fooddb
	>python queryEval.py -d 100
