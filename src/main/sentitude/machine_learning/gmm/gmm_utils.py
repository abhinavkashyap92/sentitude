import numpy as np
def mean(data_matrix):
	'''	
		This utility function calculates the mean of the data matrix 
		The data matrix on the x axis consists of a observations or data(say p observations)
		The data matrix on the y axis consists of number of samples of the data which is taken(say n samples that is taken)
		How to calculate the mean
		[1,2
		3,4
		5,6]

		mean = [1+3+5]
			   -------
			   	 3
		Similarly for the other column vector	
	'''
	return np.asmatrix(np.mean(data_matrix,axis=0))


def covariance(data_matrix):
	'''
		This utility function calculates the covariance of the data matrix
		The significance of the covariance matrix is the it shows how much overlap(covariance) in variables occur in sample
		for n * p matrix this is going to be a p * p matrix
	'''
	return np.asmatrix(np.cov(data_matrix,rowvar=0))