import numpy as np
import scipy.io.wavfile
import math
from scipy import signal
from scipy.fftpack import dct
from scipy.signal import lfilter


class mfcc:

    '''
        Mel Frequency cepstral coefficients

    '''

    def __init__(self):
        pass

    def pre_emphasis(self, signal, coefficient=0.97):
        '''
             * Passing the signal through a filter to emphasize higher frequncies of the signal
             * Increases the energy of the signal at higher frequencies
             * Formula for doing this is y[n] = x[n] - coeff * x[n-1]
         '''
        return np.append(signal[0], signal[1:] - coefficient * signal[:-1])

    def normalisation(self, signal):
        '''
            * The maximum signal amplitude is normalised to one
            * Formula for doing this is x(i) = x(i)
                                               -----     for i = 0 to N-1
                                               max(x)
        '''
        return signal / np.amax(signal)

    def framing(self, signal, frame_length, frame_step):
        '''
            * This is done to chunk the signals into small frames where each frame has adequate number of samples for analysis.
            * These are overlapping frames.
            * frame_length - the length of each frame(in milisecond)
            * frame_step - the step is the number of samples in the first frame after which the next frame should start(in milisecond)
        '''
        signal_length = len(signal)
        frame_length = round(frame_length)
        frame_step = round(frame_step)

        if signal_length < frame_length:
            number_of_frames = 1
        else:
            number_of_frames = 1 + \
                math.ceil((1 * signal_length - frame_length) / frame_step)

        padding_length = (number_of_frames - 1) * frame_step + frame_length
        number_of_zeroes_padded = padding_length - signal_length
        zero_array = np.zeros((number_of_zeroes_padded,))
        padded_signal = np.concatenate((signal, zero_array))

        # indices for framing
        indices = np.array(np.tile(np.arange(0, frame_length), (number_of_frames, 1)) +
                           np.transpose(np.tile(np.arange(0, number_of_frames * frame_step, frame_step), (frame_length, 1))), dtype=np.int32)
        return padded_signal[indices]

    def windowing(self, frames):
        '''
            * Window function is one which is zero outside a given interval
            * Why? Fourier transform assumes periodicity.
              Any discontinuity between the last sample and the repeated first sample(caused by framing) causes leakage
            * To apply the window multiply signal with window function
        '''
        return_list = []
        for frame in frames:
            return_list.append(np.multiply(frame, signal.hamming(len(frame))))

        return np.array(return_list)

    def fast_fourier_transforms(self, frames, nfft):
        '''
            * This is done to convert the signals from time domian to frequency domain
            * First apply the windowing and then multiply this with the signal
        '''
        return np.absolute(np.fft.rfft(frames, nfft))

    def power_spectrum(self, frames, nfft):
        '''
            * Calculates the power spectrum of each frame
            * power spectrum of a signal is defined as |F'( log(|F(f(t))|^2) ) |^2
            * This is useful for human voice analysis
        '''
        return 1.0 / nfft * np.square(self.fast_fourier_transforms(frames, nfft))

    def log_power_spectrum(self, frames, nfft, normalisation=1):

        power_spectrum_value = self.power_spectrum(frames, nfft)
        power_spectrum_value[power_spectrum_value <= 1e-30] = 1e-30
        log_power_spectrum = 10 * np.log10(power_spectrum_value)
        if normalisation:
            return log_power_spectrum - np.max(log_power_spectrum)
        else:
            return log_power_spectrum

    def hertz_to_mel_scale(self, hertz):
        '''
            * This is the utility function that converts the hertz to mels scale
        '''
        return 2595.0 * np.log10(1 + hertz / 700.0)

    def mel_scale_to_hertz(self, mel):
        '''
            * This is the utility function that converts mels to hertz
        '''

        return 700.00 * (10 ** (mel / 2595.0) - 1)

    def calculate_mel_filter_banks(self, num_filters=24, nfft=512, sample_rate=16000, lower_frequency=300, higher_frequency=3400):
        '''
            * This calculates the mel filter banks
            * These are triangular overlapping windows
            * Number of points required for mel scale is 2 more than the number of filters- Because it is triangular overlapping windows
            * Calculate the bin numbers
            * Calculate the filter_banks

        '''
        higher_frequency = higher_frequency or sample_rate / 2
        lower_frequency_in_mel = self.hertz_to_mel_scale(lower_frequency)
        higher_frequency_in_mel = self.hertz_to_mel_scale(higher_frequency)
        mel_scale_points = np.linspace(
            lower_frequency_in_mel, higher_frequency_in_mel, num_filters + 2)
        hertz_values = self.mel_scale_to_hertz(mel_scale_points)

        bin_numbers = np.floor((nfft + 1) * hertz_values / sample_rate)
        filter_bank = np.zeros([num_filters, nfft / 2 + 1])

        for j in xrange(0, num_filters):
            for i in xrange(int(bin_numbers[j]), int(bin_numbers[j + 1])):
                filter_bank[j, i] = (
                    i - bin_numbers[j]) / (bin_numbers[j + 1] - bin_numbers[j])
            for i in xrange(int(bin_numbers[j + 1]), int(bin_numbers[j + 2])):
                filter_bank[j, i] = (
                    bin_numbers[j + 2] - i) / (bin_numbers[j + 2] - bin_numbers[j + 1])

        return filter_bank

    def filter_bank_energies(self, filterBank, powerSpectrum):
        '''
            * Returns an array where each row  is a feature vector
            * Returns the total energy in each frame
        '''
        energy = np.sum(powerSpectrum, 1)
        feature_vectors = np.dot(powerSpectrum, np.transpose(filterBank))

        return feature_vectors, energy

    def log_filter_bank_energies(self, filterBankEnergies):
        np.seterr(divide='ignore')
        return np.log10(filterBankEnergies)

    def discrete_cosine_transforms(self, filterBankEnergies, numberCepstrals):
        return dct(filterBankEnergies, axis=1, norm='ortho')[:, :numberCepstrals]

    def lifter(self, cepstra, lifter_coeff=22):
        num_frames, number_cepstra = cepstra.shape
        n = np.arange(number_cepstra)
        lifter = 1 + (lifter_coeff / 2) * np.sin(np.pi * n / lifter_coeff)
        return lifter * cepstra

    def delta_coefficients(self, cepstrals, thetha):
        '''
            * The trajectories of the MFCC coefficients are calculated
            * This is calculated because speech has information in the dynamics
            * The formula for calculating the delta_coefficients is dt = Cn+1 - Cn-1
                                                                         -----------
                                                                               2
            * To calculate this duplicate the first and the last rows. Else it is not possible to calculate the delta vectors for the first and the last row
        '''
        # Duplicate the first and the last rows
        np.seterr(all='ignore')
        (number_rows, number_columns) = cepstrals.shape
        delta_vectors = []
        cepstrals = np.append(
            np.reshape(np.copy(cepstrals[0]), (1, number_columns)), cepstrals, axis=0)
        cepstrals = np.concatenate(
            (cepstrals, np.reshape(np.copy(cepstrals[number_rows]), (1, number_columns))), axis=0)
        for i in xrange(1, cepstrals.shape[0] - 1):
            delta_vectors.append(
                np.subtract(cepstrals[i + thetha], cepstrals[i - thetha]) / 2)

        return np.array(delta_vectors)

