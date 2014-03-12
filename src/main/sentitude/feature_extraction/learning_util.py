import numpy
def max_frames_audio(mfcc_coefficients):
	audios = mfcc_coefficients.keys()
	max_frames_audio = audios[0]
	for audio_id in audios[1:]:
		if mfcc_coefficients[audio_id][1] > mfcc_coefficients[max_frames_audio][1]:
			max_frames_audio = audio_id

	return max_frames_audio, mfcc_coefficients[max_frames_audio][1]

def preprocess_input_vectors(mfcc_vectors_dict, max_frames):
	new_mfcc_vectors_dict = {}
	for key, value in mfcc_vectors_dict.items():
		difference = max_frames - value[1]
		vector = value[0]
		if difference > 0:
			new_vector = numpy.array([[0] * 36])
			for i in range(difference):
				vector = numpy.concatenate( (vector, new_vector)) 
		new_mfcc_vectors_dict.update({key: vector})

	return new_mfcc_vectors_dict