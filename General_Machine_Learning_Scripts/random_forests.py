#############################################################################################################################################
# filename: random_forests.py
# author: Sara Davis
# date: 7/24/19
# version: 1.0
# description: This program builds a random forest model and saves it to a .dat file
#############################################################################################################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics as met
import numpy as np
import pickle
import sys
#############################################################################################################################################
# build_random_forest(X_train, y_train)
# This function builds a random forest model using x_train and y_train
# inputs: X_train- the feature vector, y_train- the label vector
# returns: clf- a random_forest model
#############################################################################################################################################
def build_random_forest(X_train, y_train, num_est, max_d):
	clf = RandomForestClassifier(n_estimators =num_est, max_depth = max_d, n_jobs = -1, random_state=0)
	clf.fit(X_train, y_train)
	return clf
############################################################################################################################################
# load_data(foldnum)
# This function loads previously constructed numpy arrays containing hog features for each image pair
# inputs: foldnum
# returns: X_train- the training feature vector, X_test- the validation feature vector, y_train - the training label vector, y_test- the testing label vector
#############################################################################################################################################
def load_data(foldnum):

	X_train = np.load("hog_numpys/fold" + str(foldnum) + "trainx.npy")
	y_train = np.load("hog_numpys/fold" + str(foldnum) + "trainy.npy")
	X_test = np.load("hog_numpys/fold" + str(foldnum) + "testx.npy")
	y_test = np.load("hog_numpys/fold" + str(foldnum) + "testy.npy")
	return X_train, y_train, X_test, y_test
############################################################################################################################################
# predictions(clf, X_test)
# This function uses the validation feature vector (X_test) and the generated random_forest model(clf) to predict the label of the validation feature vector
# inputs: row
# returns: model predictions
#############################################################################################################################################
def predictions(clf, X_test):
	return clf.predict(X_test)


#############################################################################################################################################
# calculate_truth(y_test, y_pred)
# This function uses y_test and y_pred to determine the true positive rate and true negative rate (to add more description, since accuracy is not the best metric for this)
# inputs: y_test- validation label vector y_pred- the models predicted labels
# returns: true positive rate and true negative rate
#############################################################################################################################################
def calculate_truth(y_test, y_pred):
	truePositive = 0
	trueNegative= 0
	positive = 0
	negative = 0

	for i in range(len(y_test)):
		if y_test[i] ==1:
			positive +=1
		if y_test[i] == 0:
			negative +=1
		if y_test[i] == 1 and y_test[i] == y_pred[i]:
			truePositive +=1
			print("Image Number " + str(i))
		if y_test[i] == 0 and y_test[i] ==y_pred[i]:
			trueNegative+=1
	print("TruePositive: ", str(float(truePositive)/float(positive)))
	print("TrueNegative: ", str(float(trueNegative)/float(negative)))
	return float(float(truePositive)/float(positive)), float(float(trueNegative)/float(negative))
##############################################################################################################################################
# write_to_file(y_pred, TP, TN, foldnum, acc
# This function writes true positive rate, true negative rate, accuracy, and all predictions to a file
# inputs: y_pred - predictions output by model, TP- true positive rate, TN- true negative rate, foldum- dataset fold (for naming), acc- accuracy
# returns: NONE
#############################################################################################################################################
def write_to_file(y_pred, TP, TN, foldnum, acc, n_est, max_depth):
	np.set_printoptions(threshold=sys.maxsize)
	file_to_open = "fold" + str(foldnum) + "_hog_results_RF_" + str(n_est) +"_" + str(max_depth) + ".txt"
	with open(file_to_open, "w") as file:
		
		file.write("TruePositive: ")
		file.write(str(TP))
		file.write( "\n" )
		file.write("TrueNegative: ")
		file.write(str(TN))
		file.write( "\n")
		file.write("Accuracy: ")
		file.write(str((acc)))
		file.write( "\n")
		file.write(str(y_pred))
############################################################################################################################################
# metrics(y_test, y_pred, foldnum)
# This function calculates accuracy and calls calculate_truth and write_to_file
# inputs: y_test- the validation label vector, y_pred- validation set predictions, foldnum- dataset fold (for naming
# returns: acc- the accuracy score
# NOTE: accuracy is NOT the best metric in most cases. Used here because this script was written for use by new programmers and interpretation by people with very little statistical background
#############################################################################################################################################
def metrics (y_test, y_pred, foldnum, n_est, max_depth):
	acc = met.accuracy_score(y_test, y_pred)
	TP, TN = calculate_truth(y_test, y_pred)
	print("Accuracy: ", str(acc))
	write_to_file(y_pred, TP, TN, foldnum, acc, n_est, max_depth)
	return acc
##############################################################################################################################################
# save_to_pickle(i, clf)
# This function saves the model 
# inputs: i - fold number (for naming), clf- the model
# returns: NONE
#############################################################################################################################################
def save_to_pickle(i, clf, n_est, max_depth):
	filename = "rf_hog_model_" + str(i) + "_" + str(n_est) +"_" + str(max_depth)+ ".sav"
	pickle.dump(clf, open(filename, 'wb'))
###########################################################################################################################################
# main()
# This function iterates through every fold and makes appropriate calls to generate model, test model, and save model and predictions
# also iterates through several different n_estimator values and several different max_depths to empirically find best parameters
# inputs: none
# returns: none
###########################################################################################################################################
def main():
	for i in range(1,6):
		n_estimators = [50, 100, 150, 200, 250, 300]
		for j in range(len(n_estimators)):
			max_depths=[2, 5, 10, 25, 50, 100]
			for k in range(len(max_depths)):
				X_train, y_train, X_test, y_test = load_data(i)
				clf = build_random_forest(X_train, y_train, n_estimators[j], max_depths[k])
				save_to_pickle(i, clf, n_estimators[j], max_depths[k])
				y_pred = predictions(clf, X_test)
				accuracy = metrics(y_test, y_pred, i, n_estimators[j], max_depths[k])
		
		


if __name__ =="__main__":
	main()
