import numpy
import math
import mne
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import colors

# --- Exercise 1: creating a wavelet --- #

# to make a morlet wavelet, create a gaussian and a sine wave and multiply them point by point
# base parameters
samplerate = 500   # sampling rate in Hz
wavelet_time = numpy.arange(-1, 1, 1 / samplerate)  # time, from -1 to 1 second in steps of 1/sampling-rate

# step 1: create a sine wave
# frequency of the sine wave and of gaussian in Hz = center/peak frequency of resulting wavelet
frequency = 10
# complex sine wave
sine_wave = numpy.exp(2 * numpy.pi * 1j * frequency * wavelet_time)
# plot the sine wave
fig, ax = plt.subplots(3, 1)
ax[0].plot(wavelet_time, sine_wave)
ax[0].set_title('Sine wave (signal)')
ax[0].set_ylim(-1.1, 1.1)

# step 2: make a Gaussian
# number of cycles - trade-off between temporal and frequency resolution:
# more cycles will result in higher frequency resolution, but poorer time resolution
n_cycles = 6
# standard deviation of gaussian - this will define how "wide" our wavelet is (= frequency resolution)
sigma = n_cycles / (2 * numpy.pi * frequency)
# amplitude of gaussian
amplitude = 1
gaussian_window = amplitude * numpy.exp(-wavelet_time ** 2 / (2 * sigma ** 2))
# plot gaussian
ax[1].plot(wavelet_time, gaussian_window)
ax[1].set_title('Gaussian window')

# step 3: ...and together they make a wavelet!
wavelet = gaussian_window * sine_wave
# plot wavelet
ax[2].plot(wavelet_time, wavelet)
ax[2].set_title('resulting wavelet')


# the morlet wavelet we have just created is actually complex
# plot, including the imaginary part:
fig, ax = plt.subplots(1, 1)
ax = plt.axes(projection='3d')
ax.plot(wavelet_time, wavelet.real, wavelet.imag)
ax.set_title('%i Hz complex morlet wavelet' % frequency)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('real amplitude')
ax.set_zlabel('imag amplitude')



# --- Exercise 2: convolve EEG signal with a complex morlet wavelet --- #

# load sample data  # todo this could be epochs around an auditory stimulus
fif_file = Path.cwd() / 'data' / 'experiment' / 'EEG' / 'P1' / 'Ears Free' / 'preprocessed' /'P1_Ears Free-epo.fif'
#raw = mne.io.read_raw_brainvision(header_file, preload=True)
epochs = mne.read_epochs(fname=fif_file)
data = epochs.pick_channels(['Fz'])._data[0][0]   # single channel eeg data
n_samples = len(data)  # length of the data (time-sequence)
eeg_time = numpy.arange(0, n_samples) / n_samples  # time points in the data
samplerate = epochs.info['sfreq']

# take a look at the EEG (time series) data:
fig, ax = plt.subplots(1, 1)
ax.plot(eeg_time, data)
ax.set_ylabel('Amplitude (mV)')
ax.set_xlabel('Time (s)')
ax.set_title('Epochs signal')


# step 1: define frequency range of the time-frequency-analysis (= wavelet frequencies)
min_freq = 3  # minimum frequency
max_freq = 45  # maximum frequency
n_frequencies = 20  # number of wavelets (frequency resolution)


# step 2: define wavelet parameters
# time points of the wavelet
wavelet_time = numpy.arange(-1, 1, 1 / samplerate)
# frequencies at which we will create the different wavelets (= frequency resolution)
frequencies = numpy.logspace(numpy.log10(min_freq), numpy.log10(max_freq), n_frequencies)
# number of cycles of the morlet wavelets (defined by sigma: width of the gaussian bell curve)
# note that we change the number as a function of wavelet frequency (more cycles with increasing wavelet frequency)
sigmas = numpy.logspace(numpy.log10(3), numpy.log10(10), n_frequencies) / (2 * numpy.pi * frequencies)


# step 3: set some convolution parameters
# length of the (zero-padded) wavelet in datapoints
n_wavelet = len(wavelet_time)
# length of the data in datapoints
n_data = len(data)
# length of the convolution result:
# because of  how convolution works, we end up with more datapoints than in the original signal
# visit https://www.youtube.com/watch?v=HSMwxBg7iq4&list=PLn0OLiymPak2G__qvavn3T8k7R8ssKxVr for a detailed explanation
n_convolution = n_wavelet + n_data - 1
# length of the fourier transform output; twice the length of the data due to "negative frequencies"
fft_length = 2 ** (math.ceil(math.log(n_convolution, 2)))
# half the wavelet length - used later on (to trim the edges after convolution)
half_of_wavelet_size = int((n_wavelet - 1) / 2)

# compute the fourier transform (spectral representation) of the EEG data ** needed later on for convolution
eeg_fft = numpy.fft.fft(data, fft_length)

# create numpy array to hold the results of the time-frequency analysis
eeg_power = numpy.zeros((n_frequencies, n_data))  # frequencies X time

# loop through frequencies and compute synchronization
for frequency_index in range(n_frequencies):
    # step 4: create complex morlet wavelets for each frequency:
    # frequency band specific scaling factor:
    A = numpy.sqrt(1 / (sigmas[frequency_index] * numpy.sqrt(numpy.pi)))
    # create complex sine wave at each frequency:
    sine_wave = numpy.exp(2 * 1j * numpy.pi * frequencies[frequency_index] * wavelet_time)
    # create gaussian window with increasing width (sigma)  -> more cycles with increasing wavelet frequency
    gaussian_window = numpy.exp(-wavelet_time ** 2 / (2 * (sigmas[frequency_index] ** 2)))
    # multiply to create complex morlet wavelet (cmw)
    wavelet = A * sine_wave * gaussian_window

    # step 5: convolve the eeg signal with the wavelet to retrieve time-frequency information
    # apply fourier transformation to the wavelet (convert the wavelet to frequency domain)
    wavelet_fft = numpy.fft.fft(wavelet, fft_length)
    # convolution - multiplication in the frequency domain == convolution in the time domain!
    eeg_convolved = wavelet_fft * eeg_fft  # compute the wavelet coefficients
    # -> visit https://www.youtube.com/watch?v=uSslgfKmno0 for more details
    # apply inverse fourier transformation to return to the time representation of the now convolved signal
    eeg_convolved = numpy.fft.ifft(eeg_convolved)

    # step 6: cut, convert to power and normalize
    # cut result to convolution output length (n_convolution)
    eeg_convolved = eeg_convolved[:n_convolution]
    # trim the edges (cut away the zeros that result from convolution with zero-padded / tapered gaussian window)
    eeg_convolved = eeg_convolved[half_of_wavelet_size + 1: n_convolution - half_of_wavelet_size]
    # convert amplitude to power by squaring
    time_frequency_power = (numpy.abs(eeg_convolved) ** 2)
    # baseline normalization (in this example, we use the average power of the first 500 samples as a baseline value)
    time_frequency_power_norm = 10 * numpy.log10(time_frequency_power / numpy.mean(time_frequency_power[0:50]))

    # final step: store normalized power in the time x frequency x power matrix
    eeg_power[frequency_index] = time_frequency_power_norm

# plot resulting time frequency representation of the EEG data
x, y = numpy.meshgrid(eeg_time, frequencies)
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
c_ax = ax.contour(x, y, eeg_power, linewidths=0.3, colors="k", norm=colors.Normalize())
c_ax = ax.contourf(x, y, eeg_power, norm=colors.Normalize(), cmap=plt.cm.jet)
cbar = fig.colorbar(c_ax)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
cbar.set_label('Power (dB)')
plt.show()






# extra: example of a single convolution (to show how it works)
# create wavelet
frequency = 6  # in Hz, as usual
time = numpy.arange(-1, 1, 1/samplerate)  # time vector
n_cycles = 4  # number of cycles of gaussian window - remember
sigma = (n_cycles / (2 * numpy.pi * frequency))  
wavelet = numpy.exp(2 * 1j * numpy.pi * frequency * time) * numpy.exp(-time ** 2 / (2 * sigma ** 2) / frequency)
# Fourier parameters
n_wavelet = len(wavelet)
n_data = len(data)
n_convolution = n_wavelet + n_data
half_of_wavelet_size = math.ceil((n_wavelet) / 2)
# FFT of wavelet and EEG data
fft_wavelet = numpy.fft.fft(wavelet, n_convolution)
fft_data = numpy.fft.fft(data, n_convolution)
# convolve and get inverse of fft
convolution_result_fft = numpy.fft.ifft(fft_wavelet * fft_data, n_convolution) * numpy.sqrt(sigma) # scale by root of cycles
# cut off edges
convolution_result_fft = convolution_result_fft[half_of_wavelet_size:n_convolution-half_of_wavelet_size]
# plot for comparison
fig, ax = plt.subplots(3, 1, figsize=(12,8), tight_layout=False)
ax[0].plot(time, convolution_result_fft.real)
ax[0].set_title('Projection onto real axis is filtered signal at %i Hz.'%frequency)
ax[0].set_xlabel('Time (ms)')
ax[0].set_ylabel('Voltage (\muV)')
ax[1].plot(time, numpy.abs(convolution_result_fft) ** 2)
ax[1].set_title('Magnitude of projection vector squared is power at %i Hz.'%frequency)
ax[1].set_xlabel('Time (ms)')
ax[1].set_ylabel('Voltage (\muV)')
ax[2].plot(time, numpy.angle(convolution_result_fft))
ax[2].set_title('Angle of vector is phase angle time series at %i Hz.'%frequency)
ax[2].set_xlabel('Time (ms)')
ax[2].set_ylabel('Phase angle (rad.)')




