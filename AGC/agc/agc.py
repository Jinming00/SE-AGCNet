import numpy as np 
import scipy.signal as signal  

from .utils import fft2melmx  
from .stft import stft, istft  

def tf_agc(d, sr, t_scale=1.0, f_scale=8.0, causal_tracking=True, plot=False):
    """
    Performs frequency-dependent automatic gain control on the auditory frequency axis.
    d is the input waveform (sampled at sr);
    y is the output waveform with approximately constant energy in each time-frequency region.
    t_scale is the time smoothing scale (default 1.0 sec). Controls the decay coefficient.
    f_scale is the frequency scale (default 8.0 "mel").
    causal_tracking == 0 selects traditional infinite attack, exponential release.
    causal_tracking == 1 selects symmetric, non-causal Gaussian window smoothing.
    D returns the actual STFT used in the analysis. E returns the smoothed magnitude envelope removed from D to get gain control.
    """

    hop_size = 0.032  

    ftlen = int(2 ** np.round(np.log(hop_size * sr) / np.log(2.)))  # Calculate nearest power of 2 to hop_size*sr
    winlen = ftlen  # Window length set to FFT length
    hoplen = winlen // 2  
    D = stft(d, winlen, hoplen)  # Perform Short-Time Fourier Transform on input signal
    ftsr = sr // hoplen 
    ndcols = D.shape[1]  

    nbands = max(10, 20 // f_scale)   
    mwidth = f_scale * nbands // 10  

    (f2a_tmp, _) = fft2melmx(ftlen, sr, int(nbands), mwidth)  
    f2a = f2a_tmp[:, :ftlen // 2 + 1]  # Keep only positive frequencies
    audgram = np.dot(f2a, np.abs(D))  # Calculate audiogram (mel spectrogram)

    if causal_tracking:  # If using causal tracking mode
        # Traditional attack/decay smoothing
        fbg = np.zeros(audgram.shape)  
        state = np.zeros(audgram.shape[0])  
        alpha = np.exp(-(1. / ftsr) / t_scale)  # Calculate exponential decay coefficient
        for i in range(audgram.shape[1]):  
            state = np.maximum(alpha * state, audgram[:, i]) 
            fbg[:, i] = state  

    else:
        # Non-causal, time-symmetric smoothing
        # Smooth in time with tapered window of duration ~ t_scale (uses both past and future information)
        tsd = int(np.round(t_scale * ftsr)) // 2  
     
        htlen = int(6 * tsd)  # Ensure capturing most of Gaussian window energy (6 std deviations contains >99.7%)
        twin = np.exp(-0.5 * (((np.arange(-htlen, htlen + 1)) / tsd) ** 2)).T
        # Create Gaussian window: exp(-0.5 * ((x/σ)²)) ranging from -htlen to +htlen, normalized by tsd
        AD = audgram  
        x = np.hstack((np.fliplr(AD[:, :htlen]),  # Left edge reflection: horizontally flip first htlen columns
                       AD, 
                       np.fliplr(AD[:, -htlen:]),  # Right edge reflection
                       np.zeros((AD.shape[0], htlen))))  
        fbg = signal.lfilter(twin, 1, x, 1) 
        # Remove "warmup" points, extract valid portion from filter result
        fbg = fbg[:, int(twin.size) + np.arange(ndcols)]  

    sf2a = np.sum(f2a, 0)  
    sf2a_fix = sf2a 
    sf2a_fix[sf2a == 0] = 1.  
    E = np.dot(np.dot(np.diag(1. / sf2a_fix), f2a.T), fbg)  
    # Remove any zeros in E (shouldn't happen theoretically, but just in case)
    E[E <= 0] = np.min(E[E > 0])  

    # Convert back to waveform
    y = istft(D / E, winlen, hoplen, window=np.ones(winlen))  # Use inverse STFT to get output waveform

    if plot:  
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 12))
            
            # Original STFT linear spectrogram
            plt.subplot(4, 1, 1)
            plt.imshow(20. * np.log10(np.flipud(np.abs(D))))
            plt.title('Original STFT Linear Spectrogram')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(format='%+2.0f dB')
            
            # Mel spectrogram
            plt.subplot(4, 1, 2)
            plt.imshow(20. * np.log10(np.flipud(audgram + 1e-10)), aspect='auto', 
                    interpolation='nearest', cmap='viridis')
            plt.title('Mel Spectrogram')
            plt.ylabel('Mel Band')
            plt.colorbar(format='%+2.0f dB')
            
            # Smoothed envelope linear spectrogram
            plt.subplot(4, 1, 3)
            plt.imshow(20. * np.log10(np.flipud(np.abs(E))))
            plt.title('Smoothed Envelope Spectrogram')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(format='%+2.0f dB')
            
            # Output STFT linear spectrogram
            A = stft(y, winlen, hoplen)
            plt.subplot(4, 1, 4)
            plt.imshow(20. * np.log10(np.flipud(np.abs(A))))
            plt.title('Output STFT Linear Spectrogram')
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time Frame')
            plt.colorbar(format='%+2.0f dB')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting: {e}")
            
    return (y, D, E)  # Return output waveform, input STFT and smoothed envelope