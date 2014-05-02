import mfcc
import scipy.io.wavfile
import numpy

class FeatureExtractor:

    def __init__(self,audio):
        self.audio = audio
        self.window_length = 0.02
        self.window_step = 0.01
        self.num_cepstrals = 13
        self.nfft = 512
        self.mfcc = mfcc.mfcc()

    def set_audio(self,audio):
        self.audio = audio


    def get_audio(self):
        return self.audio

    def calculate_mfcc(self):
        (self.rate,self.signal) = scipy.io.wavfile.read(self.audio)
        preEmphasizedSignal = self.mfcc.pre_emphasis(self.signal)
        normalizedSignal = self.mfcc.normalisation(preEmphasizedSignal)
        framedSignal = self.mfcc.framing(normalizedSignal,(self.rate*self.window_length),(self.rate*self.window_step))
        powerSpectrum = self.mfcc.power_spectrum(framedSignal,self.nfft)
        filterBanks = self.mfcc.calculate_mel_filter_banks(num_filters = 28,nfft=self.nfft,sample_rate = self.rate)
        (self.feature_vectors,self.energy) = self.mfcc.filter_bank_energies(filterBanks,powerSpectrum)
        self.logFilterBankEnergies = self.mfcc.log_filter_bank_energies(self.feature_vectors)
        self.mfcc_coeff_vectors = self.mfcc.lifter(self.mfcc.discrete_cosine_transforms(self.logFilterBankEnergies,self.num_cepstrals))
        self.mfcc_coeff_vectors[:,0] = numpy.log10(self.energy)
        self.delta_vectors = self.mfcc.delta_coefficients(self.mfcc_coeff_vectors,1)
        self.delta_delta_vectors = self.mfcc.delta_coefficients(self.delta_vectors,1)
        final_feature_vectors = []
        def append_delta(coeff,delta,delta_delta):
            rows = coeff.shape[0]
            for i in xrange(0,rows):
                # final_feature_vectors.append(numpy.concatenate((coeff[i],delta[i],delta_delta[i])))
                final_feature_vectors.append(numpy.concatenate((coeff[i], delta[i])))

            return numpy.array(final_feature_vectors)
        final_features = append_delta(self.mfcc_coeff_vectors,self.delta_vectors,self.delta_delta_vectors)

        # final_features = self.delta_vectors
        # final_features = self.mfcc_coeff_vectors
        ## Scikit begins ##
        # from scikits.talkbox.features import mfcc
        # final_features, mspec, spec = mfcc(self.signal)
        # final_features = self.mfcc.delta_coefficients(final_features, 1)
        ## Scikit ends ##

        final_features = final_features[~numpy.isnan(final_features).any(axis = 1)]
        final_features = final_features[~numpy.isinf(final_features).any(axis = 1)]
        final_features = final_features[~numpy.isneginf(final_features).any(axis = 1)]

        return final_features


def main():
    extractor = FeatureExtractor("/home/venkatesh/Venki/FINAL_SEM/Project/Datasets/Happiness/HappinessAudios/1.wav")
    print extractor.calculate_mfcc()[0][:13]

if __name__ == '__main__':
    main()
