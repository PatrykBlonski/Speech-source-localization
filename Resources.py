import numpy as np
import scipy.signal as ss
import scipy.linalg


# Functions to calculate LPC spectrum
def apply_filter(vad_frames):
    b, a = ss.butter(6, [0.03,0.25], 'bandpass', analog=False)
    sig=ss.lfilter(b,a,vad_frames)
    return sig 

def autocorr(vad_frames, lag=10):
        c = np.correlate(vad_frames,vad_frames, 'full')
        mid = len(c)//2
        acov = c[mid:mid+lag]
        acor = acov/acov[0]
        return(acor)
    
def LPC_coeff(vad_frames, order):
        ac = autocorr(vad_frames,order+1)
        R = scipy.linalg.toeplitz(ac[:order])
        r = ac[1:order+1]
        phi = scipy.linalg.inv(R).dot(r)
        a = np.concatenate([np.ones(1), -phi])
        return a


def LPC_freq_estimate(vad_frames,height, order=80):
    """vad_frames : vector of frames after VAD
    Returns indexes of peak frequencies for every frame in vad_frames"""

    peak_freqs = []
    sig=apply_filter(vad_frames)
    a=LPC_coeff(sig,order)
    w, h = ss.freqz(1,a,worN=640)
    h_db=10*np.log10(np.abs(h),)
    freq_index, _ = ss.find_peaks(h_db,height)  # we want frequency indexes for which fft values are at least above 15 dB
    peak_freqs.append(freq_index)

    return  np.concatenate(peak_freqs)


def tdoa(doa,microphone_array):
    
    velocity=343
    direction_vector=np.array((np.cos(doa),np.sin(doa),0))
    t_doa=np.zeros(microphone_array.shape[0])
   
    t_doa=microphone_array@direction_vector/velocity
    
    return t_doa 

def MUSIC(R, D, micarray,freq):
    
    M=micarray.shape[0] #number of microphones
    eig_val, eig_vect = np.linalg.eig(R)
    ids = np.abs(eig_val).argsort()[:(M-D)]  
    En = eig_vect[:,ids]

    peak_range=np.linspace(0,2*np.pi,360)

    i=0
    Pmusic=np.zeros(360)
    for elevation in peak_range: 
         
        sv=np.exp(-1j*2*np.pi*freq*tdoa(elevation,micarray))
        Pmusic[i] = 1/scipy.linalg.norm((sv.T@En))**2
        i+=1 
    
    return Pmusic


def r_matrix(X,peak_sum):
    
    X_k=X[:,peak_sum].reshape(8,1)
    R=X_k@X_k.conj().T    
    
    return R


def srp_phat(x, exponent, blocksize, fs):
    '''Function calculating srp-phat

    Parameters
    ----------
    x : matix [m,l]
        m - number of microphones (signals from micrphones)
    exponent : matrix
        calculated based on microphone matrix dimension, blocksize and fs
    fs : int
        sampling frequency (default: 16000 Hz)
    
    
    Returns
    --------
    array
        srp-phat (powers)
    '''
    L=x.shape[1]
    x=apply_filter(x)
    xx=x*np.hanning(L)
    X=np.fft.rfft(xx,2*L)
    start, stop = 1000*blocksize/fs,9000*blocksize/fs
    
    Y =X[:,start:stop][:,:,np.newaxis]/abs(X[:,start:stop][:,:,np.newaxis])*exponent

    YY=sum(Y,0)
    P=sum((YY*np.conjugate(YY)),0)
    
    return P

