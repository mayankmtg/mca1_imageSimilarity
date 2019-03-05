# mca1_imageSimilarity

Steps to run the code:

> python compare.py

Assumptions
1. The dataset is in the same directory
2. The format of the dataset is as follows:
	a) ./data/images/ contains the images dataset
	b) ./data/queries/ contains the query dataset
	c) ./data/relevant.txt contains information of the ground truth of the information provided
	d) ./hist_features.pck is the pickle file containing the saved extracted histogram similarity features
	e) ./kaze_features.pck is the pickle file containing the saved extracted kaze similarity features
3. All the included libraries are pre-installed on the system
4. Code uses python=version 2.7.12
5. Libraries: os, cv2, numpy, matplotlib, cPickle, scipy
