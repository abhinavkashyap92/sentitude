import numpy

def get_first_13(mfcc_coefficients):
    new_mfcc_coefficients = {}
    for audio in mfcc_coefficients.keys():
        new_mfcc_coefficients[audio] = []
        for each_vector in mfcc_coefficients[audio]:
            new_mfcc_coefficients[audio].append(each_vector[:13])
    return new_mfcc_coefficients


def get_max_frames_audio(mfcc_coefficients):
    audios = mfcc_coefficients.keys()
    max_frames_audio = audios[0]
    print max_frames_audio, mfcc_coefficients[max_frames_audio][1]
    for audio_id in audios[1:]:
        print audio_id, mfcc_coefficients[audio_id][1]
        if mfcc_coefficients[audio_id][1] > mfcc_coefficients[max_frames_audio][1]:
            max_frames_audio = audio_id

    return max_frames_audio, mfcc_coefficients[max_frames_audio][1]

def get_min_frames_audio(mfcc_coefficients):
    audios = mfcc_coefficients.keys()
    min_frames_audio = audios[0]
    for audio_id in audios[1:]:
        if mfcc_coefficients[audio_id][1] < mfcc_coefficients[min_frames_audio][1]:
            min_frames_audio = audio_id

    return min_frames_audio, mfcc_coefficients[min_frames_audio][1]


def preprocess_input_vectors(mfcc_vectors_dict, min_frames):
    new_mfcc_vectors_dict = {}
    for key, value in mfcc_vectors_dict.items():
        new_mfcc_vectors_dict.update({key: value[0][:]})
    return new_mfcc_vectors_dict
