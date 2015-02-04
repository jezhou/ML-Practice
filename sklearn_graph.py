import pylab as pl 
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn_training import *

TRAIN_SIZE = 80 / 10 # Only change the numerator

def create_line_graph(df):

	test_set = create_test_set(df)

	plots = []
	for n in [5, 10, 15, 20]:

		if n == 5: color = 'r'
		elif n == 10: color = 'c'
		elif n == 15: color = 'm'
		elif n == 20: color = 'y'

		average = []
		for seed in range(100):

			train_set = create_train_set(test_set[2], seed)

			scores = []
			for x in range(TRAIN_SIZE + 1):

				if not x: continue

				knn = train(train_set, x, n)
				scores += [accuracy_score(knn.predict(test_set[0]), test_set[1])]

			average += [scores]

		average = np.average(average, axis=0)
		temp, = pl.plot([10, 20, 30, 40, 50, 60, 70, 80], [s * 100 for s in average], color)
		plots += [temp]

	pl.ylabel('Accuracy')
	pl.xlabel('Percentage of Data Trained')
	pl.title('Accuracy vs. Sample Size - Digits')
	pl.legend(plots, ['5 neighbors', '10 neighbors', '15 neighbors', '20 neighbors'], loc=4)
	pl.show()

def create_confusion_matrix(results): 

	#extract data from tuple 
	cm = confusion_matrix(results[0], results[1])

	# Show confusion matrix in a separate window
	pl.matshow(cm)
	pl.title('Confusion matrix')
	pl.colorbar()
	pl.ylabel('True label')
	pl.xlabel('Predicted label')

	width = len(cm)
	height = len(cm[0])

	for x in xrange(width):
	    for y in xrange(height):
	        pl.annotate(str(cm[x][y]), xy=(y, x), 
	                    horizontalalignment='center',
	                    verticalalignment='center')

	pl.show()