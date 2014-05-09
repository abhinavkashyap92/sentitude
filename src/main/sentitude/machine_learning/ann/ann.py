__author__ = 'venkatesh'
import sys
import os
from collections import Counter
import operator
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from pyfann import libfann
from feature_extraction.featureExtractor import FeatureExtractor

DATASET_PATH = "../../../Dataset/"


class Ann:

    """
        Neural network class which instatiates a feed forward neural network from fann library
    """

    def __init__(self, new_network=False):
        """
            init method which initialises network parameters and setting up the network
        """
        self._connection_rate = 1
        self._learning_rate = 0.01
        self._num_input = 36
        self._num_hidden = 6
        self._num_output = 1
        self._desired_error = 0.0001
        self._max_iterations = 500
        self._iterations_between_reports = 100
        self._new_network = new_network
        self._ann = libfann.neural_net()

        if self._new_network:
            self._create_train_data()
            self._train_and_save()

    def _train_and_save(self):
        """
            method which trains the neural network
        """
        print "Training network..."
        ann = libfann.neural_net()
        ann.create_sparse_array(
            self._connection_rate, (self._num_input, self._num_hidden, self._num_output))
        ann.set_learning_rate(self._learning_rate)
        ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
        ann.train_on_file("training_data/fann/training_data.data",
                          self._max_iterations, self._iterations_between_reports, self._desired_error)
        print "Training complete..."
        print "Saving network...."
        ann.save("network_states/fann/fann.net")

    def _get_mfcc_of_training_set(self):
        """
            Reads the mffc feature vectors of the audio
        """
        print "Calculating mfcc feature vectors..."
        mfcc_coeff_vectors_dict = {}
        for i in range(1, 151):
            extractor = FeatureExtractor(
                DATASET_PATH + 'Happiness/HappinessAudios/' + str(i) + '.wav')
            mfcc_coeff_vectors = extractor.calculate_mfcc()
            mfcc_coeff_vectors_dict.update({str(i): mfcc_coeff_vectors})

        for i in range(201, 351):
            extractor = FeatureExtractor(
                DATASET_PATH + 'Sadness/SadnessAudios/' + str(i - 200) + '.wav')
            mfcc_coeff_vectors = extractor.calculate_mfcc()
            mfcc_coeff_vectors_dict.update({str(i): mfcc_coeff_vectors})
        print "mfcc feature vectors found...."
        return mfcc_coeff_vectors_dict

    def _create_train_data(self):
        """
            creates new dataset if its for new network
        """
        print "Generating training dataset..."
        mfcc_coeff = self._get_mfcc_of_training_set()
        train_data_file = open("training_data/fann/training_data.data", "w")
        train_data_file.writelines("194226 " + str(36) + " 1\n")
        for i in range(1, 151):
            for each_vector in mfcc_coeff[str(i)]:
                train_data_file.writelines(
                    (" ").join(map(str, each_vector)) + "\n")
                train_data_file.writelines("1\n")
        for i in range(201, 350):
            for each_vector in mfcc_coeff[str(i)]:
                train_data_file.writelines(
                    (" ").join(map(str, each_vector)) + "\n")
                train_data_file.writelines("0\n")

        for each_vector in mfcc_coeff[str(350)]:
            train_data_file.writelines(
                (" ").join(map(str, each_vector)) + "\n")
            train_data_file.writelines("0")

        train_data_file.close()
        print "Training data set file generated..."

    def get_emotion(self, file_path):
        """
            method to get the emotion of the given audio file
                if new_network is set then new network is created
                else previous trained network is used to guess the emotion of the audio
        """
        self._ann.create_from_file("network_states/fann/fann.net")
        extractor = FeatureExtractor(file_path)
        mfcc_vectors = extractor.calculate_mfcc()
        frame_level_values = []
        for each_vector in mfcc_vectors:
            output = self._ann.run(each_vector)
            if np.array(output) > np.array([0.5]):
                frame_level_values.append("happiness")
            else:
                frame_level_values.append("sadness")
        labels_count = Counter(frame_level_values)
        label = max(labels_count.iteritems(), key=operator.itemgetter(1))[0]
        return label


if __name__ == '__main__':
    network = Ann(new_network=True)
    print "Detected emotion: " + network.get_emotion(DATASET_PATH + 'Happiness/HappinessAudios/161.wav')
