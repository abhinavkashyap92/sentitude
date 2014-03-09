import mfcc 
import scipy.io.wavfile
import numpy

class FeatureExtractor:

	def __init__(self,audio):
		self.audio = audio
		self.window_length = 0.025
		self.window_step = 0.01
		self.num_cepstrals = 12
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
		filterBanks = self.mfcc.calculate_mel_filter_banks(num_filters = 20,nfft=self.nfft,sample_rate = self.rate)
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
				final_feature_vectors.append(numpy.concatenate((coeff[i],delta[i],delta_delta[i])))

			return numpy.array(final_feature_vectors)
		final_features = append_delta(self.mfcc_coeff_vectors,self.delta_vectors,self.delta_delta_vectors)
		final_features = final_features[~numpy.isnan(final_features).any(axis = 1)]
		final_features = final_feature[~numpy.isinf(final_features).any(axis = 1)]
		final_features = final_feature[~numpy.isneginf(final_features).any(axis = 1)]

		return final_features
		
		
def main():
	extractor = FeatureExtractor("../../audio/sample14.wav")
	print extractor.calculate_mfcc().shape

if __name__ == '__main__':
	main()
