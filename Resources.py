import numpy as np
import scipy.signal as ss
import soundfile as sf
import rir_generator as rir
from pathlib import Path
import requests
import scipy.linalg
import matplotlib.pyplot as plt 

def signals_generator(filename=None,Signal=None,velocity=343.0,fs=16000,recievers=None,source=None,Room_dimentions=None,reflection_coefficients=None,reverberation_time=None,number_of_samples=4096,order=0):
    """Returns a signal depending on the number of recievers
    https://rir-generator.readthedocs.io/_/downloads/en/latest/pdf/"""
    if filename is not None:
        signal, fs = sf.read(filename, always_2d=True)      # signal from a file and sampling frequency
    else:
        signal = Signal
        fs = fs
    if reflection_coefficients == None:
        h = rir.generate(c=velocity,fs=fs,r=recievers,s=source,L=Room_dimentions,reverberation_time=reverberation_time,nsample=number_of_samples,order=order)   # impulse responseS
    else:
        h = rir.generate(c=velocity,fs=fs,r=recievers,s=source,L=Room_dimentions,beta=reflection_coefficients,nsample=number_of_samples,order=order)   # impulse response
    microphone_signals = ss.convolve(h, signal)     # Convolve signal with impulse response

    return microphone_signals


def download_example_speach_file(add_noise = False):
    """Downloads a sample speech file for the current localization, filename is  'speech0001.wav'
       Returns data ( signal samples ) and  fs ( sampling frequency )"""
    url = "http://sp-class.agh.edu.pl/samples/speech0001.wav"

    if not isinstance('speech0001.wav', Path):
        path = Path('speech0001.wav')
    if not path.is_file():
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)

    data, fs = sf.read('speech0001.wav', always_2d=True)
    data = data / np.abs(data.max())
    if add_noise:
        noise = np.random.randn(13000)
        data = np.concatenate((noise/abs(max(noise)),data,noise/abs(max(noise))), axis=None)

    return data, fs


def VAD(sig, rate=512, overlap=256):
    """example : speech_frames=VAD(data,rate=512,overlap=256)
       rate is window size=512, overlap=256, sig=data from wavfile"""
    fs=48000
    overlapped_frames =[]
    for i in range(0, len(sig), int((overlap))):
        
        split= np.array((sig[i:i + rate]))
        # window_frame = ss.convolve(split,np.hanning(rate)) #apply window
        overlapped_frames.append(split)
        
    N_frames=len(overlapped_frames)
    #end overlap
    
    

    #calculate frame energy and zero_crossings
    energy=np.zeros(N_frames)
    Zcr=np.zeros(N_frames)
    time=np.zeros(N_frames)
    
    for i in range (N_frames):
        
        energy[i]=np.abs(1/len(overlapped_frames[i])*np.sum(((overlapped_frames[i])**2)))
        Zcr[i]=0.5*np.sum(abs(np.sign(overlapped_frames[i][1:])-np.sign(overlapped_frames[i][0:len(overlapped_frames[i])-1])))                   
   #end energy / Zcr
                
       
    #Thresholds
    E_thres=np.mean(energy)/2
    Zcr_thres=(3/2)*np.mean(Zcr)-0.3*np.std(Zcr)
    #end Thresholds
    print("E",E_thres,"Z",Zcr_thres)
    #vad decision
    vad_frames=[]
    
    for i in range(N_frames):
        if (energy[i] > E_thres and Zcr[i] < Zcr_thres):
            vad_frames.append(overlap[i])
    #end vad decision


    # # Plot detected speech -optional
    vad=np.zeros(N_frames)
    time=np.zeros(N_frames)
    for i in range(N_frames):
        time[i]=rate/2 + (i+1)*overlap
        if (energy[i] >=E_thres and Zcr[i] < Zcr_thres):
            vad[i]=max(sig)
    time_sec=np.arange(0,len(sig)/fs,1/fs)
    vad=np.interp(np.arange(len(sig)),time,vad)
    
    fig,axes=plt.subplots(1,1,figsize=(14,7))
    axes.plot(time_sec,sig)
    axes.plot(time_sec,vad,"r")
    # # end plot

    return np.array(vad_frames), E_thres, Zcr_thres  #return a vector which every element is speech frame


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

# def LPC_freq_estimate(vad_frames, order=80, height=15):
#     """vad_frames : vector of frames after VAD
#     Returns indexes of peak frequencies for every frame in vad_frames"""
#     N_frames=len(vad_frames)
#     freqs=np.linspace(0,8000,5062)
#     peak_freqs = []
#     for i in range(N_frames):
#         a=LPC_coeff(vad_frames[i],order)
#         w, h = ss.freqz(1,a)
#         h_db=10*np.log10(np.abs(h))
#         freq_index, _ = ss.find_peaks(h_db,height)  # we want frequency indexes for which fft values are at least above 15 dB
#         peak_freqs.append(freq_index)
    
#     return np.concatenate(peak_freqs).ravel()



def LPC_freq_estimate(vad_frames,height, order=80,plot=False):
    """vad_frames : vector of frames after VAD
    Returns indexes of peak frequencies for every frame in vad_frames"""
    # freqs=np.fft.fftfreq(1280,1/32000)
    # freqs=freqs[freqs>=0]

    peak_freqs = []
    sig=apply_filter(vad_frames)
    a=LPC_coeff(sig,order)
    w, h = ss.freqz(1,a,worN=640)
    h_db=10*np.log10(np.abs(h),)
    freq_index, _ = ss.find_peaks(h_db,height)  # we want frequency indexes for which fft values are at least above 15 dB
    peak_freqs.append(freq_index)
    
    #PLOT
   
    if (plot):
        fig,axes=plt.subplots(1,1,figsize=(10,10))
        #axes[0].plot(vad_frames)
        #axes[1].plot(freqs,h_db,"g")
        #axes[1].plot(freqs,10*np.log10(np.abs(np.fft.rfft(vad_frames,1023))),"k")
        #axes.plot(freqs,h_db,"g")
        #axes.set_xlim(300,3500)
        #axes[3].plot(freqs,10*np.log10(np.abs(np.fft.rfft(vad_frames,1023))),"k")
    
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


    #peak_range=np.arange(360)*np.pi/180
    peak_range=np.linspace(0,2*np.pi,360)

    # sv=np.exp(-1j*2*np.pi*1000*tdoa(peak_range,micarray))


    # Pmusic = 1/np.abs(En.T@sv)**2
    i=0
    Pmusic=np.zeros(360)
    for elevation in peak_range: 
         
        sv=np.exp(-1j*2*np.pi*freq*tdoa(elevation,micarray))
        Pmusic[i] = 1/scipy.linalg.norm((sv.T@En))**2
        i+=1 
   # Pmusic = 10* np.log10(Pmusic/np.min(Pmusic))    
     
   # doas=scipy.signal.find_peaks(Pmusic,height=7)
    
    return Pmusic


def r_matrix(X,peak_sum):
    
    X_k=X[:,peak_sum].reshape(8,1)
    R=X_k@X_k.conj().T    
    
    return R


# def matrix_R(rfft_vad, peak):
#     """ rfft_vad : fft from vad frames for each channel
#         peak : index of peak frequency from LPC spectrum
#     Returns correlation R matrix """
#     xi = np.zeros((rfft_vad.shape[0],1),dtype=complex)
#     R = np.zeros((rfft_vad.shape[0], rfft_vad.shape[0]), dtype=complex)
#     for frame in range(rfft_vad[0].shape[0]):
#         for channel in range(rfft_vad.shape[0]):
#             xi[channel] = rfft_vad[channel][frame][peak]
#         R += np.outer(xi, xi.conj())

#     return R * (1/rfft_vad[0].shape[0])   #  N =>  Number of frames   rfft_vad[0].shape[0]


def srp_phat(x, exponent):
    '''Function calculating srp-phat

    Parameters
    ----------
    x : matix [m,l]
        m - number of microphones (signals from micrphones)
    micarray : matirx [m,col]
        where col are cooridinates [x,y,z] (matrix with microphones possitions)
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
    start, stop = 20,180
    
    Y =X[:,start:stop][:,:,np.newaxis]/abs(X[:,start:stop][:,:,np.newaxis])*exponent

    YY=sum(Y,0)
    P=sum((YY*np.conjugate(YY)),0)
    
    return P


def VAD1(frame):
    """VAD dla jednej ramki, zwraca TRUE albo FALSE"""
  
    #calculate frame energy and zero_crossings
    energy=np.abs(1/len(frame)*np.sum(((frame)**2)))
    Zcr=0.5*np.sum(abs(np.sign(frame[1:])-np.sign(frame[0:len(frame)-1])))                   
   #end energy / Zcr
    #Thresholds
    E_thres=3000           #threshold without normalisation  
    Zcr_thres= 55
    if (energy > E_thres and Zcr < Zcr_thres):
        return True
    else:
        return False
    

def split_to_frames(x,L,overlap):
    '''Dzieli sygnał na ramki'''
    frames =[]
    frames2 =[]
    frames3 =[]
    frames4 =[]

    for i in range(0, x.shape[1]-500, int((overlap))):

        split= np.array((x[0,i:i + L]))
        split2= np.array((x[1,i:i + L]))
        split3= np.array((x[2,i:i + L]))
        split4= np.array((x[3,i:i + L]))        # window_frame = ss.convolve(split,np.hanning(rate)) #apply window

        frames.append(split)
        frames2.append(split2)
        frames3.append(split3)
        frames4.append(split4)

    return np.stack((frames,frames2,frames3,frames4),1)

# Variables
velocity = 343.0                 # Sound velocity (m/s)
microphone_array = np.array([                     # Receiver position(s) [x y z] (m)
        [1.0, 1.0, 1.0],
        [0.1, 1.0, 1.0],
        [1.0, 0.1, 1.0]])

array_mat_mm =np.array([
   [ 0.00,     0.00,  0.00],
   [-38.13,    3.58,  0.00],
   [-20.98,   32.04,  0.00],
   [ 11.97,   36.38,  0.00],
   [ 35.91,   13.32,  0.00],
   [ 32.81,  -19.77,  0.00],
   [ 5.00,   -37.97,  0.00],
   [-26.57,  -27.58,  0.00]])
array_mat_mm*=0.001




source_position = [5.0, 5.0, 1.8]       # Source position [x y z] (m)
room_dimensions = [6.0, 6.0, 3.0]       # Room dimensions [x y z] (m)
beta=[0.3, 0.2, 0.5, 0.1, 0.1, 0.1]  # example of reflection coefficients



#EXAMPLE OF USING RIR_generator

# download_example_speach_file()
# #reverberation_time=0.4  # Reverberation time (s) is required if beta = None
# nsample=4096            # Number of output samples
# # Matrix of three signals
# microphone_signals = signals_generator(filename="speech0001.wav", velocity=velocity, recievers=microphone_array, source=source_position, Room_dimentions=room_dimensions, reflection_coefficients=beta, number_of_samples=nsample)
# print(microphone_signals.shape)



