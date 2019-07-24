###################################################################################################################################################
# filename: crop_images.py
# author: Sara Davis
# date: 7/15/19
# version: 2.0
# description: This program accesses every image in the caricature dataset and allows images to be cropped, resized, and viewed
# note: To work as is, must be saved to RET_2019/Facial_Landmarking/Face_Bounding_Boxes 
###################################################################################################################################################


from PIL import Image
from PIL import ImageOps
import numpy as np
import cv2 as cv
import os
import csv


####################################################################################
# get_files()
# This function gets a list of all files in the parent directry
# inputs: none
# returns: a list of all files in the parent directory
###################################################################################
def get_files():
	curDir = os.getcwd()
	PATH_TO_IMAGES = "Face_Bounding_boxes/data"
	dir_tree_images = os.path.join(curDir, PATH_TO_IMAGES)
	dir_tree_bboxes = curDir
	#Get the name of each file in the directory
	files = [f for f in os.listdir(curDir) if os.path.isfile(f)]
	files = sorted(files)
	return files


####################################################################################
# adjust_vals(xmin, ymin, xmax, ymax, w, h)
# This function adjusts a crop to the nearest multiple of 178 * 218
# inputs: xmin, ymin, xmax, ymax, w, h
# returns: adjusted xmin, ymin, xmax, ymax
###################################################################################


def adjust_vals( xmin, ymin, xmax, ymax, w, h):
	cropW = xmax - xmin
	cropH = ymax - ymin
	if cropW < 89 and cropH < 109:
		rw = (89 - (cropW % 89)) /2
		rH = (109 - (cropH % 109))/2
		xmin = xmin -rw
		xmax = xmax + rw
		ymin = ymin - rH
		ymax = ymax + rH
	elif cropW % 178 != 0 or cropH%218 != 0:
		xmult = int(cropW / 178) + 1
		ymult = int (cropH /218) + 1
		if xmult > ymult:
			rw = int((178 - (cropW % 178)) /2)
			xmin = xmin - rw
			xmax = xmax + rw
			rh = int((218 - (cropH %218))/2)
			ymin = ymin - rh
			ymax = ymax + rh
			diffY = ymax - ymin
			val = 218 * xmult
			ymin = ymin -int(( abs(diffY-val)/2))
			ymax = ymax + int((abs(diffY-val)/2))
		if ymult > xmult:
			rh =int( (218 - (cropH %218))/2)
			ymin = ymin - rh
			ymax = ymax + rh
			rw = int((218 - (cropW % 218)) /2)
			xmin = xmin - rw
			xmax = xmax + rw
			diffX = xmax - xmin
			val = 178 * ymult
			xmin = xmin -int((abs(diffX - val)/2))
			xmax = xmax -int((abs(diffX -val) /2))
		if xmult == ymult:
			rh = int((218 - (cropH %218))/2)
			ymin = ymin - rh
			ymax = ymax + rh
			rw = int((178 - (cropW % 178)) /2)
			xmin = xmin - rw
			xmax = xmax + rw
			diffX = xmax - xmin
	if xmin < 0:
		xmin = 1
	if ymin < 0:
		ymin=1
	if xmax > w:
		xmax = w-1
	if ymax > h:
		ymax = h-1
	return xmin, ymin, xmax, ymax


####################################################################################
#save_dir_crop(n, file_to_open, suffix, string)
# This function creates a save path for cropped images
# inputs: n (filename), file_to_open(file path), any appended suffixes (a-f), and string (image name)
# returns: adjusted xmin, ymin, xmax, ymax
###################################################################################

def save_dir_crop(n, file_to_open, suffix, string):
	ext = ".JPEG"

	saveDir = os.path.join(os.getcwd(), "data/", file_to_open)
	saveDir = saveDir+"/cropped/"
	if not os.path.exists(saveDir):
		os.makedirs(saveDir)
	if suffix is "None":
		p = os.path.join(saveDir,  n + string)
	else:
		p = os.path.join(saveDir, n + suffix +string)
	return p

####################################################################################
# get_suffix(image)
# This function finds any appended suffixes to names (a-f)
# inputs: image
# returns: im, suffix
###################################################################################

def get_suffix(image):
	im = image
	suffix = "None"
	if "a" in image or "b" in image or "c" in image or "d" in image or "e" in image or "f" in image:
		if "a" in image:
			suffix = "a"
		if "b" in image:
			suffix = "b"
		if "c" in image:
			suffix = "c"
		if "d" in image:
			suffix = "d"
		if "e" in image:
			suffix = "e"
		if "f" in image:
			suffix = "f"
		im = image[:-5]
		im = im + ".jpg"
	return im, suffix

####################################################################################
# get_information(row)
# This function takes information from txt file row, and populates xmin, ymin, xmax, ymax, and image name
# inputs: row
# returns: xmin, ymin, xmax, ymax, imagename
###################################################################################
def get_information(row):
	image = row[0]
	xmin = float(row[1])
	ymin = float(row[2])
	xmax = float(row[3])
	ymax = float(row[4])
	return image, xmin, ymin, xmax, ymax

####################################################################################
# crop()
# This function uses get_information, get_suffix, adjust_vals, and save_dir to open all images in subdirectories, crop them, and save them
# inputs: None
# returns: None
###################################################################################
def crop():
	files = get_files()
	#Open each text file and gets the image name, xmin, ymin, xmax, ymax
	for f in files:
		if f.endswith(".txt") and "Readme" not in f:
			file_to_open = f[:-9]
			print(file_to_open)
			with open(f) as csvfile:
				filereader = csv.reader(csvfile, delimiter= ' ')
				for row in filereader:
					image, xmin, ymin, xmax, ymax = get_information(row)
					#Split the a b and c of for multiface images, so the image can be located and saved
					image, suffix = get_suffix(image)					
					#open image, get its size
					im = Image.open( os.path.join(os.getcwd(), "data/", file_to_open, image)).convert('RGB')
					w, h = im.size
					print(image)
					#Find the number of pixels necessary to expand the box to some multiple of 178 *218 and crops it
					xmin, ymin, xmax, ymax = adjust_vals(xmin, ymin, xmax, ymax, w, h)
					crop = im.crop((xmin, ymin, xmax, ymax))	
					#Save to the correct directory
					n = image[:-4]
					path = save_dir_crop(n, file_to_open, suffix, "_crop.jpg")
					crop.save(path)
####################################################################################
# resize()
# This function uses get_information to open all images in subdirectories, resize them, and save them
# inputs: None
# returns: None
###################################################################################			

def resize():	
	files = get_files()
	#Open each text file and gets the image name, xmin, ymin, xmax, ymax
	for f in sorted(files):
		if f.endswith(".txt") and "Readme" not in f:
			file_to_open = f
			with open(f) as csvfile:
				filereader = csv.reader(csvfile, delimiter= ' ')
				for row in filereader:
					image, xmin, ymin, xmax, ymax = get_information(row)
					#Split the a b and c of for multiface images, so the image can be located and saved
					image = image[:-4] + "_crop.jpg"
					im = Image.open( os.path.join(os.getcwd(), "data/", file_to_open[:-9],"cropped", image)).convert('RGB')			
					#define a path, resize, save
					p = os.path.join(os.getcwd(), "data/", file_to_open[:-9], "cropped", image[:-4] + "_resized.jpg")
					im =im.resize((178, 218), Image.ANTIALIAS)
					im.save(p)

####################################################################################
# check_crop
# This function uses get_information to open all resized images in subdirectories for viewing
# inputs: None
# returns: None
###################################################################################
def check_crop():	
	files = get_files()
	print(files)
	#Open each text file and gets the image name, xmin, ymin, xmax, ymax
	for f in files:
		if f.endswith(".txt") and "Readme" not in f:
			file_to_open = f
			print(file_to_open)
			with open(f) as csvfile:
				filereader = csv.reader(csvfile, delimiter= ' ')
				for row in filereader:
					image, xmin, ymin, xmax, ymax = get_information(row)
					image = image[:-4] + "_crop_resized.jpg"
					print(image)
					imageDir = os.path.join(os.getcwd(), "data/", file_to_open[:-9],"cropped", image)
					im = cv.imread(imageDir)
					cv.imshow(imageDir, im)
					cv.waitKey(0)
					cv.destroyAllWindows()



crop()
resize()
#check_crop()
