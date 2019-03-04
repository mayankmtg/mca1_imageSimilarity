# from skimage.measure import structural_similarity as ssim
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def compare_images_ssim(imageA, imageB):
    s = ssim(imageA, imageB)
    return s



def getGroundTruth(imagename):
    search = imagename.split('.')[0]
    with open('data/relevant.txt') as f:
        for line in f:
            if(search == line.split(':')[0]):
                return line.split(':')[1].rstrip('\n')
        
    return None
# img1 = cv2.imread("img1.jpg")
# img2 = cv2.imread("img2.jpg")

# print compare_images(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY))
# print compare_images(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY))

for query in os.listdir('data/queries'):
	if(query[0]=='.'):
		continue
	queryfilename = 'data/queries/'+ query
	query_cv2 = cv2.imread(queryfilename,0)
	query_cv2 = cv2.resize(query_cv2, (240,160))
	# print query_cv2.shape
	similar_set = []
	for data in os.listdir('data/images'):
		if(data[0]=='.'):
			continue
		datafilename = 'data/images/'+ data
		data_cv2 = cv2.imread(datafilename,0)
		data_cv2 = cv2.resize(data_cv2, (240,160))
		# print data_cv2.shape
		similarity = ssim(query_cv2, data_cv2)
		similar_set.append((data.split('.')[0], similarity))
	similar_set.sort(key=lambda x:x[1], reverse=True)
	# print similar_set
	ground_truth = np.array(getGroundTruth(query).split(','))
	# for img in groun
	compare_truth = []
	for i in range(len(ground_truth)):
		compare_truth.append(similar_set[i][0])
	compare_truth = np.array(compare_truth)
	accuracy_score = str(len(np.intersect1d(ground_truth, compare_truth))) + '/' + str(len(compare_truth))
	print accuracy_score

