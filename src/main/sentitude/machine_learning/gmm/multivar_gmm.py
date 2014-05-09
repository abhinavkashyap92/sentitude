import numpy as np
import test
from gmm_utils import mean
from gmm_utils import covariance


class MultiVariateGaussian():
	'''
		This represents the Multivariate Gaussian Distribution or the multivariate Normal Distribution
		The probability density function of Multivariate Normal distribution is given as 
	'''


	def __init__(self,mean_vector,covariance_matrix,number_of_random_variables):
		'''
				mean_vector: numpy matrix
							 The mean vector specifies the mean of the random variables with a dimension of p*1 for a data matrix of dimension n*p
				covariance_matrix: numpy array
									The covariance matrix is an array of covariances and relative covariance of size p * p for a data matrix of dimension n*p
									We would be interested in the diagonal elements of the covariance matrix
				number_of_random_variables: number
									The number of random variables specifies the number of gaussian clusters that is considered							 
		'''
		self.number_of_random_variables = number_of_random_variables
		self.mean_vector = mean_vector
		self.covariance_matrix = np.diag(np.diag(covariance_matrix))
		self.generalisedSampleVariance = np.linalg.det(self.covariance_matrix) 
		self.precision_matrix = np.fabs(np.linalg.inv(self.covariance_matrix))
		
	def getNumberOfRandomVariables(self):
		return self.number_of_random_variables

	def setNumberOfRandomVariables(self,number_of_random_variables):
		self.number_of_random_variables = number_of_random_variables

	def getMeanVector(self):
		return self.mean_vector

	def setMeanVector(self,mean_vector):
		self.mean_vector = mean_vector

	def getCovarianceMatrix(self):
		return self.covariance_matrix

	def setCovarianceMatrix(self,covariance_matrix):
		self.covariance_matrix = covariance_matrix

	def getGeneralisedSampleVariance(self):
		return self.generalisedSampleVariance;

	def setGeneralisedSampleVariance(self):
		self.generalisedSampleVariance = np.linalg.det(self.getCovarianceMatrix()) 

	def getPrecisionMatrix(self):
		return self.precision_matrix;

	def setPrecisionMatrix(self):
		self.precision_matrix = np.fabs(np.linalg.inv(self.getCovarianceMatrix()))

	def probabilityDensityFunction(self,data):
		variation_from_mean = data - self.mean_vector
		self.MahalanobisDistance =  ((variation_from_mean) * self.precision_matrix ) * (variation_from_mean.T)
		self.denominator = ((2*np.pi) ** (self.number_of_random_variables/2)) * (self.generalisedSampleVariance ** 0.5)
		return (np.exp((self.MahalanobisDistance * (-0.5) )) / self.denominator).item(0)

if __name__ == '__main__':
	data = test.get_test_data()
	mean = mean(data)
	covariance = covariance(data)
	print "covariance_matrix",covariance
	gmm_cluster = MultiVariateGaussian(mean, covariance,2);
	print gmm_cluster.probabilityDensityFunction(np.matrix([3.0,4.0]))


	