from Pycluster import kcluster as knn
import test
from gmm_utils import mean
from gmm_utils import covariance
from multivar_gmm import MultiVariateGaussian
import numpy as np
import sys


class GMM():

    """
                This is to model the 'Gaussian mixture model'
                Consider a vector x = {x1,x2,x3.....xn} and every vector is in a d dimensional space
                Let there be m number of clusters and each cluster represented by the letter k
                Consider the probability distribution function of a cluster k
                Then the gaussian mixture model is
                                p = Sum(wkpk) for k from 1 to m
                                        here wk is the weight of the kth mixture
                                        The sum of all the weights has to be 1
                                        p is the probability distribution of the mixture of gaussians
                                        pk is the Multivariate gaussian probability density function modelled in multivar_gmm.py
                                        (thetha for every component of Gaussian is the mean vector, covariance matrix)

    """

    def __init__(self, data, number_of_gaussians):
        """
                        data: a nd array
                        number_of_gaussians: number
                                                                        The number of clusters to which the data is divided
                        gaussians: list
        """
        self.number_of_gaussians = number_of_gaussians
        self.data = data
        self.gaussians = []
        self.initialisation()
        self.mstep(self.estep())

    def getGaussians(self):
        return self.gaussians

    def initialisation(self):
        """
                        This is the initialisation of the parameters of the gaussians
                        On the choice of using the Pycluster library see http://stackoverflow.com/questions/1545606/python-k-means-algorithm
        """
        labels, errors, found = knn(self.data, self.number_of_gaussians)
        # grouped data example
        # 0th element of grouped data represents the 0th cluster and consists
        # of the frames belonging to that cluster
        group_data = [[] for i in xrange(self.number_of_gaussians)]
        for index, label in enumerate(labels):
            group_data[label].append(self.data[index].tolist())

        total_weight = sum(map(lambda x: len(x), group_data))

        dimension = (self.data.shape)[1]
        for everyMatrix in group_data:
            everyMatrix = np.array(everyMatrix)
            mean_vector = mean(everyMatrix)
            covariance_matrix = covariance(everyMatrix)
            gaussian = MultiVariateGaussian(
                mean_vector, covariance_matrix, dimension)
            weight = float(len(everyMatrix)) / total_weight
            self.gaussians.append({'gaussian': gaussian, 'weight': weight})

    def estep(self, data=None):
        """
                        In the expectation step of the EM algorithm find the probability p(i,k)
                        The probability of the sample i belonging to the cluster k using the available mixture parameter estimates

                        alpha(k,i) = pi(k)pdf(x;ak,sk) -> a is the mean vector and s is the covariance matrix
                                                -------------------
                                                pi(j)pdf(x:aj,sj) summation over j = 1 to m

                        see the formula on open cv page http://docs.opencv.org/modules/ml/doc/expectation_maximization.html
        """
        if data == None:
            data = self.data
        else:
            data = data

        number_clusters = self.number_of_gaussians
        size = len(data)

        # constructing a two d array where every row(i) i from 1 to n is for one vector
        # and column contains the value of the probability of that vector
        # belonging to that cluster k = 1 to m
        alpha_array = np.zeros([size, number_clusters])

        for i in xrange(size):
            for k in xrange(number_clusters):
                pi_k = self.gaussians[k]["weight"]
                pdf_k = self.gaussians[k][
                    "gaussian"].probabilityDensityFunction(np.asmatrix(data[i]))
                alpha_array[i][k] = pi_k * pdf_k

            alpha_array[i] = alpha_array[i] / np.sum(alpha_array[i])

        return alpha_array

    def mstep(self, alpha_array):
        """
                This is the maximisation step: In this step use the probability that is calculated in the estep to re estimate the parameters for all the clusters
                pi(k) = 1
                           -- alpha[i][k] summation over i from 1 to N
                            N

                a(k) = alpha[i][k]*x[i] summation over i from 1 to N
                           ----------------
                                alpha[i][k] 	summation over i from 1 to N

                s(k) = alpha[i][k]*(x(i) - a(k))*(x(i) - a(k)).T summation over i from 1 to N
                           -----------------------------------------
                             alpha[i][k] 							 summation over i from 1 to N

                Calculations are shown for just one of the clusters
                Do this for all the clusters from 1 to m
        """
        size = len(self.data)

        def summation_alpha_i_k(cluster_number):
            """
                    to calculate sigma(alpha[i][k]) for a given k over i = 1 to N
            """
            return np.sum(alpha_array[:, cluster_number])

        def summation_alpha_i_k_into_x_i(cluster_number):
            """	to calculate sigma(alpha[i][k]*x[i]) for a given k over i = 1 to N
            """
            alpha_array_transpose = alpha_array.T
            k_alpha_array = alpha_array_transpose[cluster_number]
            return sum([k_alpha_array[i] * self.data[i] for i in xrange(size)])

        def calculating_numerator_for_s_k(cluster_number):
            alpha_array_transpose = alpha_array.T
            k_alpha_array = alpha_array_transpose[cluster_number]
            return sum([k_alpha_array[i] * np.multiply((self.data[i] - self.gaussians[cluster_number]["gaussian"].getMeanVector()), (self.data[i] - self.gaussians[cluster_number]["gaussian"].getMeanVector()).T) for i in xrange(size)])

        for k in xrange(self.number_of_gaussians):
            alpha_i_k_summation = summation_alpha_i_k(k)
            pi_k = alpha_i_k_summation / size
            a_k = np.asmatrix(
                summation_alpha_i_k_into_x_i(k) / float(alpha_i_k_summation))
            s_k = np.asmatrix(
                np.diag(np.diagonal(calculating_numerator_for_s_k(k))) / float(alpha_i_k_summation))
            # update the parameters of the corresponding gaussian
            self.gaussians[k]["weight"] = pi_k
            self.gaussians[k]["gaussian"].setMeanVector(a_k)
            self.gaussians[k]["gaussian"].setCovarianceMatrix(s_k)
            self.gaussians[k]["gaussian"].setPrecisionMatrix()
            self.gaussians[k]["gaussian"].setGeneralisedSampleVariance()

    def likelihood(self, data):
        """
                The definition of a likelihood function is given at http://en.wikipedia.org/wiki/Likelihood_function#Continuous_probability_distribution
                The likelihood of the gaussian mixture model is given in http://docs.opencv.org/modules/ml/doc/expectation_maximization.html

        """
        size = len(data)
        # Dont worry this line came after a lot of refractoring
        # I agree its nowhere readable
        return sum([np.log(sum([self.gaussians[k]["weight"] * self.gaussians[k]["gaussian"].probabilityDensityFunction(np.asmatrix(self.data[i])) for k in xrange(self.number_of_gaussians)])) for i in xrange(size)])

    def train(self, niter=20):
        """
                        The expectation maximisation algorithm tries to maximise the maximum likelihood function of the mixture of gaussians iteratively
                        Look at http://parkcu.com/blog/wp-content/uploads/2013/07/Bradley-Fayyad-Reina-1998-Scaling-EM-Expectation-Maximization-Clustering-to-Large-Databases.pdf to know  how it works
        """
        likelihoods = list()
        for i in xrange(niter):
            alpha_array = self.estep()
            self.mstep(alpha_array)
            likelihoods.append(self.likelihood(self.data))
            if len(likelihoods) >= 2 and (abs(likelihoods[-1] - likelihoods[-2]) < abs(1e-5)):
                print "breaking at iteration i: ", i
                break

    def predict(self, data):
        """
                This method predicts the data label to which the sample belongs.
                On how to do this is got from the paper http://www.eecs.tufts.edu/~mcao01/2010f/COMP-135.pdf
        """
        alpha_matrix = self.estep(data=data)
        res = alpha_matrix.argmax(axis=1)
        unique_vals, indices = np.unique(res, return_inverse=True)
        return unique_vals[np.argmax(np.bincount(indices))]


if __name__ == '__main__':
    data = test.get_test_data()
    gmm = GMM(data, 2)
