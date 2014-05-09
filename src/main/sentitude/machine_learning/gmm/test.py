import numpy as np
import os
import sys
import gmm
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from feature_extraction.featureExtractor import FeatureExtractor


def get_test_data():
    return np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])


class TestGMM(object):

    def __init__(self, number_of_samples):
        self.number_of_samples = number_of_samples
        self.happy_features = self.get_audio_frames(
            "../../../Dataset/Happiness/HappinessAudios")
        self.sadness_features = self.get_audio_frames(
            "../../../Dataset/Sadness/SadnessAudios")
        self.train(np.vstack((self.happy_features, self.sadness_features)))
        self.test()

    def get_audio_frames(self, file_path):
        all_frames = []
        for i in xrange(1, self.number_of_samples):
            extractor = FeatureExtractor(file_path + "/" + str(i) + ".wav")
            all_frames.append(extractor.calculate_mfcc())

        return np.vstack(tuple(all_frames))

    def train(self, data):
        self.gmm_object = gmm.GMM(data, 2)
        self.gmm_object.train()

    def test(self):
        sadness_correctCount = 0
        for i in xrange(150, 200):
            extractor = FeatureExtractor(
                "../../../Dataset/Sadness/SadnessAudios/" + str(i) + ".wav")
            input_frames = extractor.calculate_mfcc()
            if self.gmm_object.predict(input_frames) == 1:
                sadness_correctCount = sadness_correctCount + 1

        print "sadness correct count: ", sadness_correctCount

        happiness_correctCount = 0
        for i in xrange(150, 200):
            extractor = FeatureExtractor(
                "../../../Dataset/Happiness/HappinessAudios/" + str(i) + ".wav")
            input_frames = extractor.calculate_mfcc()
            if self.gmm_object.predict(input_frames) == 0:
                happiness_correctCount = happiness_correctCount + 1

        print "happiness correct count: ", happiness_correctCount


if __name__ == '__main__':
    test = TestGMM(150)
