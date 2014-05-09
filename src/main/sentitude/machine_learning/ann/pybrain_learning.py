import sys
import os
from collections import Counter
import operator

from pybrain.datasets.classification import ClassificationDataSet
from pybrain.datasets.classification import SupervisedDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, FeedForwardNetwork,\
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
        extractor = FeatureExtractor('/home/venkatesh/Venki/FINAL_SEM/Project/Datasets/Happiness/HappinessAudios/' + str(i) + '.wav')
        mfcc_coeff_vectors = extractor.calculate_mfcc()
        mfcc_coeff_vectors_dict.update({str(i): (mfcc_coeff_vectors, mfcc_coeff_vectors.shape[0])})

    for i in range(201, 401):
        extractor = FeatureExtractor('/home/venkatesh/Venki/FINAL_SEM/Project/Datasets/Sadness/SadnessAudios/' + str(i - 200) + '.wav')
        mfcc_coeff_vectors = extractor.calculate_mfcc()
        mfcc_coeff_vectors_dict.update({str(i): (mfcc_coeff_vectors, mfcc_coeff_vectors.shape[0])})

    audio_with_min_frames, min_frames = get_min_frames_audio(mfcc_coeff_vectors_dict)
    processed_mfcc_coeff = preprocess_input_vectors(mfcc_coeff_vectors_dict, min_frames)
    frames = min_frames
    print "mfcc found...."
    classes = ["happiness", "sadness"]
    try:
        network = NetworkReader.readFrom('network_state_new_.xml')
    except:
        # Create new network and start Training
        training_data = ClassificationDataSet(frames * 26, target=1, nb_classes=2, class_labels=classes)
        # training_data = SupervisedDataSet(frames * 39, 1)
        for i in range(1, 151):
            mfcc_coeff_vectors = processed_mfcc_coeff[str(i)]
            training_data.appendLinked(mfcc_coeff_vectors.ravel(), [1])
            # training_data.addSample(mfcc_coeff_vectors.ravel(), [1])

        for i in range(201, 351):
            mfcc_coeff_vectors = processed_mfcc_coeff[str(i)]
            training_data.appendLinked(mfcc_coeff_vectors.ravel(), [0])
            # training_data.addSample(mfcc_coeff_vectors.ravel(), [0])

        training_data._convertToOneOfMany()
        network = buildNetwork(training_data.indim, 5, training_data.outdim)
        trainer = BackpropTrainer(network, learningrate=0.01, momentum=0.99)
        print "Before training...", trainer.testOnData(training_data)
        trainer.trainOnDataset(training_data, 1000)
        print "After training...", trainer.testOnData(training_data)
        NetworkWriter.writeToFile(network, "network_state_new_.xml")

    print "*" * 30 , "Happiness Detection", "*" * 30
    for i in range(151, 201):
        output = network.activate(processed_mfcc_coeff[str(i)].ravel())
        # print output,
        # if output > 0.7:
        #     print "happiness"
        class_index = max(xrange(len(output)), key=output.__getitem__)
        class_name = classes[class_index]
        print class_name

    # print "*" * 30 , " Sadness Detection", "*" * 30
    # for i in range(351, 401):
    #     output = network.activate(processed_mfcc_coeff[str(i)].ravel())
    #     # print output,
    #     # if output < 0.3:
    #         # print "sadness"
    #     class_index = max(xrange(len(output)), key=output.__getitem__)
    #     class_name = classes[class_index]
    #     print class_name

if __name__ == '__main__':
    main()
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % ( 0.5, 500))