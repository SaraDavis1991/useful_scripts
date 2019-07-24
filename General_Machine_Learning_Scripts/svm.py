#############################################################################################################################################
# filename: svm.py
# author: Sara Davis
# date: 7/24/19
# version: 1.0
# description: This program builds an svm model and saves it to a .dat file
#############################################################################################################################################
from sklearn import svm
from sklearn import metrics as met
import numpy as np
import pickle
import sys
#############################################################################################################################################
# build_svm(X_train, y_train)
# This function builds a random forest model using x_train and y_train
# inputs: X_train- the feature vector, y_train- the label vector
# returns: clf- an svm model
#############################################################################################################################################
def build_svm(X_train, y_train):
	clf = svm.SVC(kernel='rbf')
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
def write_to_file(y_pred, TP, TN, foldnum, acc):
	np.set_printoptions(threshold=sys.maxsize)
	file_to_open = "fold" + str(foldnum) + "_hog_results_SVM.txt"
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
def metrics (y_test, y_pred, foldnum):
	acc = met.accuracy_score(y_test, y_pred)
	TP, TN = calculate_truth(y_test, y_pred)
	print("Accuracy: ", str(acc))
	write_to_file(y_pred, TP, TN, foldnum, acc)
	return acc
##############################################################################################################################################
# save_to_pickle(i, clf)
# This function saves the model 
# inputs: i - fold number (for naming), clf- the model
# returns: NONE
#############################################################################################################################################
def save_to_pickle(i, clf):
	filename = "svm_hog_model_" + str(i) +".sav"
	pickle.dump(clf, open(filename, 'wb'))

###########################################################################################################################################
# main()
# This function iterates through every fold and makes appropriate calls to generate model, test model, and save model and predictions
# inputs: none
# returns: none
###########################################################################################################################################
def main():
	for i in range(1,6):
		X_train, y_train, X_test, y_test = load_data(i)
		clf = build_svm(X_train, y_train)
		save_to_pickle(i, clf)
		y_pred = predictions(clf, X_test)
		accuracy = metrics(y_test, y_pred, i)
		
		


if __name__ =="__main__":
	main()
