###################################################################################################################################################
# filename: open_folds.py
# author: Sara Davis
# date: 7/15/19
# version: 1.0
# description: This program opens a fold list of people, opens the associated image pairing fold csv for both training and validation, and opens the csv file containing the hog features for the images in each pairing. Finally, it calculates
# description, cont: the absolute difference between the pair, and adds it to numpy array (train or test) . Also grabs match label from csv and adds to numpy array of labels (train or test)
################################################################################################################################################
import csv
import numpy as np 
import os
from scipy.spatial.distance import cdist
import pandas as pd
import random
#create global variable since onesCount and zerosCount need to be incremented in multiple functions, but passing only allows a shallow copy
onesCount = 0
zerosCount = 0
########################################################################################################################################################################
# get_information(row)
# This function takes information from txt file row, and populates xmin, ymin, xmax, ymax, and image name
# inputs: row
# returns: xmin, ymin, xmax, ymax, imagename
#######################################################################################################################################################################
def get_information(row):
	imagepath1 = row[0]

	imagepath2 = row[1]
	if len(row) == 4:
		label = int(row[2])
	else:
		label = 0
	
	return imagepath1, imagepath2, label
######################################################################################################################################################################
# split_image_and_name(image_and_name)
# This function splits the image and file directory names from the csv file 
# inputs: image_and_name
# returns: image, celeb_dir
#####################################################################################################################################################################
def split_image_and_name(image_and_name):
	counter = 0
	letter_counter = 0
	if ".jpg" not in image_and_name:
		image_and_name = image_and_name + ".jpg"
	dirName= image_and_name[:-7]
	imageName = image_and_name.replace(dirName, '')
	
	if dirName.endswith('_'):
		dirName = dirName[:-1]
	if imageName.startswith('_'):
		imageName = imageName[1:]
	
	
	return(imageName, dirName)
######################################################################################################################################################################
# generate_random_list(rows, idxList, data, portion)
# This function uses recursion to generate a random int that represents the row to access. Compares zerosCount and onesCount to ensure that they remain balanced in train set (since entire data set is imbalanced)
# inputs: rows- num rows in csv, idxList- indexes to gather features from, data-pandas dataframe with all information, portion- the number of allowable values in the idxList
# returns: generate_random_list if value already exists in the idxList. Otherwise, it returns the label for that row
#####################################################################################################################################################################
def generate_random_list(rows, idxList, data, portion):
	global zerosCount
	global onesCount	
	val = random.randint(0, rows)
	
	if val not in idxList:
		if data.iloc[val][2] == 0 and zerosCount <= int(portion/2):
					
			zerosCount +=1
			return val
		elif data.iloc[val][2] == 1 and onesCount <= int(portion/2):			
			onesCount +=1			
			return val
	return generate_random_list(rows, idxList, data, portion)

######################################################################################################################################################################
# main()
# Opens txt as csv and reads the rows to get image name, directory, and feature vector. Calls other functions. Portions dataset
# inputs: None
# returns: None
#####################################################################################################################################################################
def main():
	
	cwd = os.getcwd()
	
	print('1')
	#open the fold file
	path = os.path.join(cwd, "folds", "fold1_train.txt") 
	data = pd.read_csv(path, delimiter = ' ')
	rows, cols = data.shape
	print(rows, cols)
	idxList = []
	labelList = []
	portion = int (rows / 50)
	
	for i in range(portion):
		val = generate_random_list(rows-1, idxList, data, portion)
		idxList.append(val)
	
	print("zeros" + str(zerosCount))
	print("ones" + str(onesCount))
	

	with open(path) as csvfile:
		filereader = csv.reader(csvfile, delimiter = " ")
		X_train = np.empty((portion, 200))
		
		y_train = np.zeros((portion))

		
		#x_train and y_train were not the same size when np.empty was used on y_train
		
		count = 0
		for idx, row in enumerate(filereader):
			if idx in idxList:
				#get each image name/directory
				imagepath1, imagepath2, label = get_information(row) 
				labelList.append(int(label))
				image1name, image1dir = split_image_and_name(imagepath1)
				image2name , image2dir = split_image_and_name(imagepath2)
				
				


				#open the feature extraction file with the directory name and find the image name in the file, grab its features
				path2 = os.path.join(cwd, "Facial_Landmarking", "Face_Bounding_Boxes", "hog", image1dir + ".csv" )	
				with open(path2) as csvfile2:
					filereader2=csv.reader(csvfile2, delimiter=" ")
					for row in filereader2:
						#print(len(row))
						if row[0]==image1name:
							featuresim1=row[1:]
							#print(featuresim1)
			
				#open the feature extraction file with the directory name and find the image name in the file, grab its features
				path3 = os.path.join(cwd, "Facial_Landmarking", "Face_Bounding_Boxes", "hog" , image2dir + ".csv" )
				with open(path3) as csvfile2:
					filereader2=csv.reader(csvfile2, delimiter=" ")
					for row in filereader2:
						if row[0]==image2name:
							featuresim2=row[1:]
							#print(featuresim2)
						print(image2dir, image2name)
								
				#convert features1 and features2 to ints from strings
				featuresim1=map(float, featuresim1)
				featuresim2 = map(float, featuresim2)
				image1array = np.array(featuresim1)
				image2array = np.array(featuresim2)
				imsdiffts= np.absolute(image1array - image2array)
				X_train[count] = imsdiffts
				count +=1
						
			
	y_train = np.asarray(labelList)
	
	print("x " , X_train.shape)
	print("y ", y_train.shape)
	np.save('hog_numpys/fold1trainx.npy', X_train)
	np.save('hog_numpys/fold1trainy.npy', y_train)


	path = os.path.join(cwd, "folds", "fold1_val.txt") 
	data = pd.read_csv(path)
	rows, cols = data.shape
	idxList = []
	labelList = []
	
	idxList= list(range(0, rows))
	
	with open(path) as csvfile:
		
		filereader = csv.reader(csvfile, delimiter = " ")
		X_test = np.empty((rows, 200))
		
		count = 0
		for idx, row in enumerate(filereader):
			if idx in idxList:
				#get each image name/directory
				imagepath1, imagepath2, label = get_information(row) 
				
				image1name, image1dir = split_image_and_name(imagepath1)
				image2name , image2dir = split_image_and_name(imagepath2)
			
				#open the feature extraction file with the directory name and find the image name in the file, grab its features
				path2 = os.path.join(cwd, "Facial_Landmarking", "Face_Bounding_Boxes", "hog", image1dir + ".csv" )	
				with open(path2) as csvfile2:
					filereader2=csv.reader(csvfile2, delimiter=" ")
					for row in filereader2:
						#print(len(row))
						if row[0]==image1name:
							featuresim1=row[1:]
							#print(featuresim1)
			
				#open the feature extraction file with the directory name and find the image name in the file, grab its features
				path3 = os.path.join(cwd, "Facial_Landmarking", "Face_Bounding_Boxes", "hog" , image2dir + ".csv" )
				with open(path3) as csvfile2:
					filereader2=csv.reader(csvfile2, delimiter=" ")
					for row in filereader2:
						if row[0]==image2name:
							featuresim2=row[1:]
							#print(featuresim2)
			
				#convert features1 and features2 to ints from strings
				featuresim1=map(float, featuresim1)
				featuresim2 = map(float, featuresim2)
				image1array = np.array(featuresim1)
				image2array = np.array(featuresim2)
				imsdiffts= np.absolute(image1array - image2array)
				X_test[count] = imsdiffts
				labelList.append(label)
				count +=1

	y_test = np.asarray(labelList)	
	print("x ", X_test.shape)
	print("y", y_test.shape)
	
	np.save('hog_numpys/fold1testx.npy', X_test)
	np.save('hog_numpys/fold1testy.npy', y_test)
	
if __name__ =="__main__":
	main()

