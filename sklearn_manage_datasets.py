import numpy as np

from sklearn import datasets
from pandas import DataFrame, read_csv

def create_iris():
	#Load the raw iris data. Features are transposed so that features are on the
	#row instead of each column. This is easier to extract one by one.
	raw_data = datasets.load_iris()
	features = raw_data.data.transpose()
	labels   = raw_data.target

	sepal_length = features[0].transpose()
	sepal_width  = features[1].transpose()
	petal_length = features[2].transpose()
	petal_width  = features[3].transpose()

	#Use pandas to convert raw_data into a csv file. Export the csv file.
	IrisDataSet = zip(labels, sepal_length, sepal_width, petal_length, petal_width)
	initial_df = DataFrame(data = IrisDataSet, columns = ['Sample Name', 'S. Length', 'S. Width', 'P. Length', 'P. Width'])
	initial_df.to_csv('irisData.csv', index = False, header = True)

def get_iris():
	#Use pandas to covert csv file into raw_data
	return read_csv('irisdata.csv')

def create_digits(): 
	raw_data = datasets.load_digits()
	features = raw_data.images
	labels = raw_data.target

	n_samples = len(features)
	flat_data = features.reshape((n_samples, -1))

	labels_with_features = np.insert(flat_data, 0, labels, axis=1)
	DataFrame(labels_with_features).to_csv('digitsData.csv', index = False)

def get_digits():
	return read_csv('digitsData.csv')
