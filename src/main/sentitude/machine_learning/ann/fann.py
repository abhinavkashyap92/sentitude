__author__ = 'venkatesh'
import sys
import os
from collections import Counter
import operator
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from pyfann import libfann
from feature_extraction.featureExtractor import FeatureExtractor
from learning_util import get_min_frames_audio, preprocess_input_vectors

def main():
    mfcc_coeff_vectors_dict = {}
    for i in range(1, 201):
        extractor = FeatureExtractor('../../../Dataset/Happiness/HappinessAudios/' + str(i) + '.wav')
        mfcc_coeff_vectors = extractor.calculate_mfcc()
        mfcc_coeff_vectors_dict.update({str(i) :(mfcc_coeff_vectors, mfcc_coeff_vectors.shape[0])})

    for i in range(201, 401):
        extractor = FeatureExtractor('../../../Dataset/Sadness/SadnessAudios/' + str(i - 200) + '.wav')
        mfcc_coeff_vectors = extractor.calculate_mfcc()
        mfcc_coeff_vectors_dict.update({str(i) :(mfcc_coeff_vectors, mfcc_coeff_vectors.shape[0])})

    processed_mfcc_coeff = preprocess_input_vectors(mfcc_coeff_vectors_dict, 0)

    # Prepare training dataset
    # train_data_file = open("training_data/fann/training_data.data", "w")
    # train_data_file.writelines("194226 " + str(36) + " 1\n")
    # for i in range(1, 151):
    #     for each_vector in processed_mfcc_coeff[str(i)]:
    #         train_data_file.writelines((" ").join(map(str, each_vector)) + "\n")
    #         train_data_file.writelines("1\n")
    #
    # for i in range(201, 350):
    #     for each_vector in processed_mfcc_coeff[str(i)]:
    #         train_data_file.writelines((" ").join(map(str, each_vector)) + "\n")
    #         train_data_file.writelines("0\n")
    #
    # for each_vector in processed_mfcc_coeff[str(350)]:
    #     train_data_file.writelines((" ").join(map(str, each_vector)) + "\n")
    #     train_data_file.writelines("0")
    #
    # train_data_file.close()
    #
    # print "Data prepared...."
    #
    # connection_rate = 1
    # learning_rate = 0.01
    # num_input = 36
    # num_hidden = 6
    # num_output = 1
    #
    # desired_error = 0.0001
    # max_iterations = 500
    # iterations_between_reports = 100
    # #
    # ann = libfann.neural_net()
    # ann.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
    # ann.set_learning_rate(learning_rate)
    # ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
    #
    # ann.train_on_file("training_data/fann/training_data.data", max_iterations, iterations_between_reports, desired_error)
    #
    # ann.save("network_states/fann/fann.net")
    # print "done!"
    # Create neural network from file
    ann = libfann.neural_net()
    ann.create_from_file("network_states/fann/fann.net")  # A trained network ll be loaded

    # Test for happiness detection
    print "*" * 30, "Happiness Detection", "*" * 30
    counter = {
        'happiness': 0,
        'sadness': 0
    }
    for i in range(151, 201):
        mfcc_coeff_vectors = processed_mfcc_coeff[str(i)]
        frame_level_values = []
        for each_vector in mfcc_coeff_vectors:
            output = ann.run(each_vector)
            if np.array(output) > np.array([0.5]):
                frame_level_values.append("happiness")
            else:
                frame_level_values.append("sadness")
        labels_count = Counter(frame_level_values)
        label = max(labels_count.iteritems(), key=operator.itemgetter(1))[0]
        print str(i) + ".wav: " + label
        counter[label] = counter[label] + 1
    print
    print counter
    print
    #
    # # This is test for sadness detection
    print "*" * 30, "Sadness Detection", "*" * 30
    counter = {
        'happiness': 0,
        'sadness': 0
    }

    for i in range(351, 401):
        mfcc_coeff_vectors = processed_mfcc_coeff[str(i)]
        frame_level_values = []
        for each_vector in mfcc_coeff_vectors:
            output = ann.run(each_vector)
            if np.array(output) > np.array([0.5]):
                frame_level_values.append("happiness")
            else:
                frame_level_values.append("sadness")
        labels_count = Counter(frame_level_values)
        label = max(labels_count.iteritems(), key=operator.itemgetter(1))[0]
        print str(i - 200) + ".wav: " + label
        counter[label] = counter[label] + 1
    print
    print counter

# def get_sentiment(file):
#     ann = libfann.neural_net()
#     ann.create_from_file("fann_try_15000.net")
#     extractor = FeatureExtractor(file)
#     mfcc_coeff_vectors_dict = {}
#     frame_level_values = []
#     mfcc_coeff_vectors = extractor.calculate_mfcc()
#     mfcc_coeff_vectors_dict.update({'audio' :(mfcc_coeff_vectors, mfcc_coeff_vectors.shape[0])})
#     processed_mfcc_coeff = preprocess_input_vectors(mfcc_coeff_vectors_dict, 0)
#     for each_vector in processed_mfcc_coeff['audio']:
#         output = ann.run(each_vector)
#         if np.array(output) > np.array([0.5]):
#             frame_level_values.append("happiness")
#         else:
#             frame_level_values.append("sadness")
#     labels_count = Counter(frame_level_values)
#     label = max(labels_count.iteritems(), key=operator.itemgetter(1))[0]
#     return label

if __name__ == '__main__':
    main()