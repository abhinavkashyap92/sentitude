import sys
import os

from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tests.helpers import gradientCheck

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from feature_extraction.featureExtractor import FeatureExtractor
from learning_util import max_frames_audio, preprocess_input_vectors

def main():
	mfcc_coeff_vectors_dict = {}
	for i in range(1, 9):
		extractor = FeatureExtractor('../../audio/'+ str(i) + '.wav')
		mfcc_coeff_vectors = extractor.calculate_mfcc()
		mfcc_coeff_vectors_dict.update({str(i) :(mfcc_coeff_vectors, mfcc_coeff_vectors.shape[0])})

	audio_with_max_frames , frames = max_frames_audio(mfcc_coeff_vectors_dict)
	processed_mfcc_coeff = preprocess_input_vectors(mfcc_coeff_vectors_dict, frames)
	data_set = SupervisedDataSet(frames * 36, 1)
	for i in range(1, 9):
		if i == 5:
			continue
		mfcc_coeff_vectors = processed_mfcc_coeff[str(i)]
		vector = []
		for each_vector in mfcc_coeff_vectors:
			vector.extend(list(each_vector))
		data_set.addSample(vector, [1])
	
	extractor = FeatureExtractor('../../audio/voice.wav')
	mfcc_coeff_vectors = extractor.calculate_mfcc()
	processed_mfcc_coeff = preprocess_input_vectors({'voice':(mfcc_coeff_vectors, mfcc_coeff_vectors.shape[0])}, frames)
	
	vector = []
	for each_vector in processed_mfcc_coeff['voice']:
		vector.extend(list(each_vector))
	data_set.addSample(vector, [0])

	extractor = FeatureExtractor('../../audio/5.wav')
	mfcc_coeff_vectors = extractor.calculate_mfcc()
	processed_mfcc_coeff = preprocess_input_vectors({'5':(mfcc_coeff_vectors, mfcc_coeff_vectors.shape[0])}, frames)
	test_vector = []
	for each_vector in processed_mfcc_coeff['5']:
		test_vector.extend(list(each_vector))
	
	network = buildNetwork(frames * 36, 10, 1)
	trainer = BackpropTrainer(network, learningrate=0.01, momentum=0.99)
	print 'MSE before', trainer.testOnData(data_set)
	trainer.trainOnDataset(data_set, 500) # Train for 500 epoch - this is giving best results right now
	print "MSE after", trainer.testOnData(data_set)
	print network.activate(test_vector) # Value close to 1 or more than 1 says it has laughter

if __name__ == '__main__':
	main()
