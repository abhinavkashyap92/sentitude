import sys
import os
from collections import Counter
import operator

from pybrain.datasets.classification import ClassificationDataSet
from pybrain.datasets.classification import SupervisedDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, FeedForwardNetwork, \
    FullConnection
from pybrain.tools.customxml import NetworkReader, NetworkWriter

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from feature_extraction.featureExtractor import FeatureExtractor
from learning_util import get_max_frames_audio, preprocess_input_vectors, \
    get_first_13, get_min_frames_audio


def main():
    print "Calculating mfcc...."
    mfcc_coeff_vectors_dict = {}
    for i in range(1, 201):
        extractor = FeatureExtractor(
            '/home/venkatesh/Venki/FINAL_SEM/Project/Datasets/Happiness/HappinessAudios/' + str(i) + '.wav')
        mfcc_coeff_vectors = extractor.calculate_mfcc()
        mfcc_coeff_vectors_dict.update({str(i): (mfcc_coeff_vectors, mfcc_coeff_vectors.shape[0])})

    for i in range(201, 401):
        extractor = FeatureExtractor(
            '/home/venkatesh/Venki/FINAL_SEM/Project/Datasets/Sadness/SadnessAudios/' + str(i - 200) + '.wav')
        mfcc_coeff_vectors = extractor.calculate_mfcc()
        mfcc_coeff_vectors_dict.update({str(i): (mfcc_coeff_vectors, mfcc_coeff_vectors.shape[0])})

    audio_with_min_frames, min_frames = get_min_frames_audio(
        mfcc_coeff_vectors_dict)
    processed_mfcc_coeff = preprocess_input_vectors(
        mfcc_coeff_vectors_dict, min_frames)
    # frames = min_frames
    # print frames
    # print len(processed_mfcc_coeff['1'])
    # for each_vector in processed_mfcc_coeff['1']:
    #     print len(each_vector)
    print "mffcc found..."
    classes = ["happiness", "sadness"]

    training_data = ClassificationDataSet(
        26, target=1, nb_classes=2, class_labels=classes)
    # training_data = SupervisedDataSet(13, 1)
    try:
        network = NetworkReader.readFrom(
            'network_state_frame_level_new2_no_pp1.xml')
    except:
        for i in range(1, 51):
            mfcc_coeff_vectors = processed_mfcc_coeff[str(i)]
            for each_vector in mfcc_coeff_vectors:
                training_data.appendLinked(each_vector, [1])

        for i in range(201, 251):
            mfcc_coeff_vectors = processed_mfcc_coeff[str(i)]
            for each_vector in mfcc_coeff_vectors:
                training_data.appendLinked(each_vector, [0])

        training_data._convertToOneOfMany()
        print "prepared training data.."
        print training_data.indim, training_data.outdim
        network = buildNetwork(
            training_data.indim, 5, training_data.outdim, fast=True)
        trainer = BackpropTrainer(network, learningrate=0.01, momentum=0.99)
        print "Before training...", trainer.testOnData(training_data)
        trainer.trainOnDataset(training_data, 1000)
        print "After training...", trainer.testOnData(training_data)
        NetworkWriter.writeToFile(
            network, "network_state_frame_level_new2_no_pp.xml")

    # print "*" * 30 , "Happiness Detection", "*" * 30
    # for i in range(1, 11):
    #     mfcc_coeff_vectors = processed_mfcc_coeff[str(i)]
    #     frame_level_values = []
    #     for each_vector in mfcc_cCounteroeff_vectors:
    #         output = network.activate(each_vector)
    #         print output
    #         class_index = max(xrange(len(output)), key=output.__getitem__)
    #         frame_level_values.append(classes[class_index])
    #     labels_count = Counter(frame_level_values)
    #     print labels_count
        # print max(labels_count.iteritems(), key=operator.itemgetter(1))[0]
    # output = network.activate(processed_mfcc_coeff[str(i)].ravel())
    # print output,
    # if output > 0.7:
    # print "happiness"

if __name__ == '__main__':
    main()
