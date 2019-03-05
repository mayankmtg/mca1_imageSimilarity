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
			data_cv2 = cv2.imread(datafilename,0)
			data_cv2 = cv2.resize(data_cv2, (240,160))
			self.dataDataset.append([str(data), data_cv2])
			# (data_filename, data_cv2_object)
		# self.dataDataset = np.array(self.dataDataset)

	def extract(self, vectorSize = 32):
		alg = cv2.KAZE_create()
		self.extractedFeatures = {}
		for data_filename, data_cv2_object in self.dataDataset:
			keypoints = alg.detect(data_cv2_object)
			keypoints = sorted(keypoints, key=lambda x: -x.response)[:vectorSize]
			keypoints, dsc = alg.compute(data_cv2_object, keypoints)
			dsc = dsc.flatten()
			needed_size = (vectorSize * 64)
			if dsc.size < needed_size:
				dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
			self.extractedFeatures[data_filename] = dsc

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

	def compare_ground_truth(self,queryFileName, similar_array):
		groundTruth = self.get_ground_truth(queryFileName)
		groundTruth = np.array(groundTruth.split(','))
		# print groundTruth
		compareTruth = []
		similar_array.sort(key=lambda x:x[1], reverse=True)
		for i in range(len(groundTruth)):
			compareTruth.append(similar_array[i][0])
		compareTruth = np.array(compareTruth)
		accuracy_score = (len(np.intersect1d(groundTruth, compareTruth)),len(compareTruth))
		return accuracy_score



# extract_feature = Extract_Feature()
# extract_feature.extract()
# extract_feature.save()

extractedFeatures = pickle.load(open('features.pck', 'r'))
# print len(extractedFeatures.keys())
compare_image = Compare_Image(extractedFeatures)
sum1=0
sum2=0
for query in os.listdir(query_dataset):
	if(query[0]=='.'):
		continue
	query_similarity = compare_image.cosine_similarity(query_dataset + '/' + query)
	query_similarity = sorted(query_similarity, key=lambda x:float(x[1]), reverse=True)
	# print query_similarity
	# TODO: change type to float of the second column of query similarity
	s1,s2 = compare_image.compare_ground_truth(query, query_similarity)
	sum1+=s1
	sum2+=s2
print float(sum1)/float(sum2)

