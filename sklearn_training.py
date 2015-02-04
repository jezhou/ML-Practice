import numpy as np
import pandas as pd

from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 20
TRAIN_SIZE = 80 / 10 # Only change the numerator

def create_test_set(dataframe):

	#Create DataFrame / Series subsets. labels_df at this point is a series, features_df is a DF
	labels_df = dataframe.iloc[:, 0]
	features_df = dataframe.iloc[:,1:]

	#Shuffle and set the partition size
	np.random.seed(0)
	indices = np.random.permutation(len(features_df.index))
	testsize = get_size(len(features_df.index), TEST_SIZE)

	iris_features_test  = features_df.iloc[indices[-testsize:],:].as_matrix()
	iris_labels_test    = labels_df.iloc[indices[-testsize:]].as_matrix()

	corrected_features = features_df.iloc[indices[:-testsize],:]
	corrected_labels   = DataFrame(labels_df.iloc[indices[:-testsize]])
	dataframe = pd.concat([corrected_labels, corrected_features], axis=1)

	return (iris_features_test, iris_labels_test, dataframe)

def create_train_set(dataframe, train_gen_seed=0):

	dataframe.reset_index(drop=True, inplace=True)

	#Create DataFrame / Series subsets. labels_df at this point is a series, features_df is a DF
	labels_df = dataframe.iloc[:, 0]
	features_df = dataframe.iloc[:,1:]

	# Shuffle everything
	np.random.seed(train_gen_seed)
	indices = np.random.permutation(len(features_df.index))
	features_train = features_df.iloc[indices,:]
	labels_train   = labels_df.iloc[indices]

	#Get random samples from 0 to length - 10 for label and feature training.
	#Also, split equally into X parts so we can train data part by part.
	#Get random samples from length - 10 to length for verification.
	features_train = np.array_split(features_train.as_matrix(), TRAIN_SIZE)
	labels_train   = np.array_split(labels_train.as_matrix(), TRAIN_SIZE)

	return (features_train, labels_train)

def train(train_tuple, parts_to_train, n_neighbors):

	train_features, train_labels = train_tuple[0], train_tuple[1]

	#Get N parts we want to train. The more parts we get, the more data
	#we are using to train the data.
	features_train = np.concatenate(train_features[:parts_to_train])
	labels_train = np.concatenate(train_labels[:parts_to_train])

	#Time to use machine learning on the data we have collected. 
	knn = KNeighborsClassifier(n_neighbors=n_neighbors)
	knn.fit(features_train, labels_train)

	#format the data we just collected, return as knn object
	return knn

def get_size(num_indices, percentage):
	return int(num_indices * percentage / 100.0)