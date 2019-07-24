###################################################################################################################################################
# filename: check_landmarks.py
# author: Sara Davis
# date: 7/15/19
# version: 1.0
# description: This program opens bad_facial_landmarks and the associated image so that landmarks can be cleaned up
################################################################################################################################################

from PIL import Image
from PIL import ImageOps
import numpy as np
import cv2 as cv
import os
import csv
import subprocess, sys


########################################################################################################################################################################
# get_information(row)
# This function takes information from txt file row, and populates xmin, ymin, xmax, ymax, and image name
# inputs: row
# returns: xmin, ymin, xmax, ymax, imagename
#######################################################################################################################################################################
def get_information(row):
	image_and_name = row[0]

	landmarkList = row[1]
	
	return image_and_name, landmarkList

######################################################################################################################################################################
# open_files()
# This function opens the good facial landmarks and bad facial landmarks so we can edit the bad facial landmark, and then copy/paste into the good facial landmarks
# inputs: none
# returns: none
#####################################################################################################################################################################
def open_files():
	opener = "open" if sys.platform == "darwin" else "xdg-open"
	subprocess.call([opener, "failed_facial_landmarks_coordinates.txt"])
	subprocess.call([opener, "facial_landmarks_coordinates.txt"])
######################################################################################################################################################################
# string_to_list(landmarkList)
# This function cleans the string read from csv file, and turns it into a list of x,y coordinate lists
# inputs: landmarkList
# returns: lists
#####################################################################################################################################################################
def string_to_list(landmarkList):
	landmarkList = landmarkList.replace(']', '],')
	landmarkList = landmarkList[:-3]
	landmarkList = landmarkList + "]"
	strs= landmarkList.replace('[','').split('],')
	lists = [map(int, s.replace(']','').split(',')) for s in strs]
	return lists
######################################################################################################################################################################
# split_image_and_name(image_and_name)
# This function splits the image and file directory names from the csv file 
# inputs: image_and_name
# returns: image, celeb_dir
#####################################################################################################################################################################
def split_image_and_name(image_and_name):
	image = image_and_name[:3]
	length = 4
			
	if "_" in image:
		image = image[:-1]
		length = 3
			
	celeb_dir = image_and_name[length:]
	return image, celeb_dir
######################################################################################################################################################################
# plot_on_image(path_to_image, lists, image_and_name)
# This function plots the facial landmarks on the image so we can find the coordinates a landmark SHOULD be at
# inputs: path_to_image, lists, image_and_name
# returns: none
#####################################################################################################################################################################	
def plot_on_image(path_to_image, lists, image_and_name):
	im = cv.imread(path_to_image)
	for i in range(len(lists)):
		x = lists[i][0]
		y = lists[i][1]
		cv.circle(im, (x,y), 2, (0, 255, 0), -1)		
	cv.imshow(image_and_name, im)
	cv.waitKey(0)
			
######################################################################################################################################################################
# main()
# Opens txt as a csv and reads the rows to get the image name, directory, and facial landmarks stored in failed_facial_landmarks. Calls all other funcitons
# inputs: none
# returns: none
#####################################################################################################################################################################
def main():
	open_files()
	with open("failed_facial_landmarks_coordinates.txt") as csvfile:
		filereader = csv.reader(csvfile, delimiter= ' ')
		for row in filereader:
			image_and_name, landmarkList = get_information(row)
			lists = string_to_list(landmarkList)	
			image, celeb_dir =split_image_and_name(image_and_name)
			
			path_to_image = os.path.join("Face_Bounding_Boxes", "data", celeb_dir,"cropped", image + "_crop_resized.jpg")

			plot_on_image(path_to_image, lists, image_and_name)



if __name__ =="__main__":
	main()
