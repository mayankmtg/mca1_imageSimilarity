import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from scipy.spatial.distance import cdist as cDist

query_dataset = "data/queries"
data_dataset = "data/images"

class Extract_Feature:
	def __init__(self):
		self.dataDataset = []
		for data in os.listdir(data_dataset):
			if(data[0]=='.'):
				continue
			datafilename = data_dataset + '/' + data
			data_cv2 = cv2.imread(datafilename)
			data_cv2 = cv2.resize(data_cv2, (240,160))
			self.dataDataset.append([str(data), data_cv2])
			# (data_filename, data_cv2_object)
		# self.dataDataset = np.array(self.dataDataset)

	def extract_kaze(self, vectorSize = 32):
		alg = cv2.KAZE_create()
		self.extractedFeatures = {}
		for data_filename, data_cv2_object in self.dataDataset:
			data_cv2_object = cv2.cvtColor(data_cv2_object, cv2.COLOR_BGR2GRAY)
			keypoints = alg.detect(data_cv2_object)
			keypoints = sorted(keypoints, key=lambda x: -x.response)[:vectorSize]
			keypoints, dsc = alg.compute(data_cv2_object, keypoints)
			dsc = dsc.flatten()
			needed_size = (vectorSize * 64)
			if dsc.size < needed_size:
				dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
			self.extractedFeatures[data_filename] = dsc

	def extract_histogram(self):
		self.extractedFeatures = {}
		for data_filename, data_cv2_object in self.dataDataset:
			data_cv2_object_RGB = cv2.cvtColor(data_cv2_object, cv2.COLOR_BGR2RGB)
			hist = cv2.calcHist([data_cv2_object], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
			hist = cv2.normalize(hist, hist).flatten()
			self.extractedFeatures[data_filename] = hist

	def save(self, features_pck = "features.pck"):
		with open(features_pck, 'w') as fp:
			pickle.dump(self.extractedFeatures, fp)

class Compare_Image:
	
	def __init__(self, extractedFeatures):
		# self.extractedFeatures = extractedFeatures
		self.data_filenames = []
		self.data_features = []
		for name,imgFeature in extractedFeatures.iteritems():
			self.data_filenames.append(name.split('.')[0])
			self.data_features.append(imgFeature)

	def get_ground_truth(self, queryImageName):
		search = queryImageName.split('.')[0]
		with open('data/relevant.txt') as f:
			for line in f:
				if(search == line.split(':')[0]):
					return line.split(':')[1].rstrip('\n')
		return None
	
	def cosine_similarity(self, queryFileName, vectorSize = 32):
		self.queryImage = cv2.imread(queryFileName, 0)
		self.queryImage = cv2.resize(self.queryImage, (240,160))
		alg = cv2.KAZE_create()	
		keypoints = alg.detect(self.queryImage)
		keypoints = sorted(keypoints, key=lambda x: -x.response)[:vectorSize]
		keypoints, dsc = alg.compute(self.queryImage, keypoints)
		dsc = dsc.flatten()
		needed_size = (vectorSize * 64)
		if dsc.size < needed_size:
			dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
		v = dsc.reshape(1, -1)
		cosine_dist = cDist(self.data_features, v, 'cosine').reshape(-1)
		return np.array((self.data_filenames, cosine_dist)).T
	
	def histogram_similarity(self, queryFileName, histogramFeatures):
		self.queryImage = cv2.imread(queryFileName)
		self.queryImage = cv2.resize(self.queryImage, (240,160))
		hist = cv2.calcHist([self.queryImage], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
		hist = cv2.normalize(hist, hist).flatten()
		retArray = []
		for data_filename, histF in histogramFeatures.iteritems():
			distance = cv2.compareHist(hist, histF, cv2.HISTCMP_BHATTACHARYYA)
			retArray.append([data_filename, distance])
		return np.array(retArray)

		

	def compare_ground_truth(self,queryFileName, similar_array):
		groundTruth = self.get_ground_truth(queryFileName)
		groundTruth = np.array(groundTruth.split(','))
		# print groundTruth
		compareTruth = []
		similar_array.sort(key=lambda x:x[1], reverse=True)
		similar_array = similar_array[:20]
		for i in range(len(similar_array)):
			compareTruth.append(similar_array[i][0])
		compareTruth = np.array(compareTruth)
		accuracy_score = len(np.intersect1d(groundTruth, compareTruth)) / float(len(compareTruth))
		return accuracy_score



extract_feature = Extract_Feature()

print "Extraction Kaze"
# extract_feature.extract_kaze()
# extract_feature.save("kaze_features.pck")

print "Extraction Hist"
# extract_feature.extract_histogram()
# extract_feature.save("hist_features.pck")

print "Extraction Complete"


extractedFeatures = pickle.load(open('kaze_features.pck', 'r'))
histogramFeatures = pickle.load(open('hist_features.pck', 'r'))

# print len(extractedFeatures.keys())
compare_image = Compare_Image(extractedFeatures)
sum1=0
sum2=0
sumd=0
no_of_queries = 0
for query in os.listdir(query_dataset):
	if(query[0]=='.'):
		continue
	query_similarity_kaze = compare_image.cosine_similarity(query_dataset + '/' + query)
	query_similarity_kaze = sorted(query_similarity_kaze, key=lambda x:x[0])
	# print query_similarity_kaze
	# TODO: change type to float of the second column of query similarity

	query_similarity_hist = compare_image.histogram_similarity(query_dataset + '/' + query, histogramFeatures)
	query_similarity_hist = sorted(query_similarity_hist, key=lambda x:x[0])

	w_kaze = 50
	w_hist = -50
	query_similarity_concat = []
	# print len(query_similarity_hist), len(query_similarity_kaze)
	for i in range(len(query_similarity_kaze)):
		query_similarity_concat.append([query_similarity_kaze[i][0], float(query_similarity_kaze[i][1]) * w_kaze + float(query_similarity_hist[i][1]) * w_hist])

	query_similarity_concat = sorted(query_similarity_concat, key=lambda x:x[1], reverse=True)
	# query_similarity_concat = np.array(query_similarity_concat)
	s1 = compare_image.compare_ground_truth(query, query_similarity_concat)
	# sum1+=s1
	# sum2+=s2
	# print s1
	sumd+=float(s1)

	no_of_queries+=1
print no_of_queries
print sumd / no_of_queries
