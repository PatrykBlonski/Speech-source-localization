#!/usr/bin/env python
#Program estimating DOA (destination of arrival) for speech sources using MUSIC and SRP-PHAT.
#Devices used: raspberry pi, microphone matrix
#authors:
#Patryk Błoński
#Bartłomiej Woś
#Bartosz Kawa
#date: 20.01.2023


import socket
from webrtcvad import Vad
from Resources import srp_phat,MUSIC,r_matrix,LPC_freq_estimate #module with all necessary functions
import matplotlib.pyplot as plt
import numpy as np

micarray = np.array([               #microphone matrix used in project
   [ 0.00 ,    0.00 , 0.00],
   [-38.13,    3.58 , 0.00],
   [-20.98,   32.04 , 0.00],
   [ 11.97,   36.38 , 0.00],
   [ 35.91,   13.32 , 0.00],
   [ 32.81,  -19.77 , 0.00],
   [ 5.00 ,  -37.97 , 0.00],
   [-26.57,  -27.58 , 0.00]
])
micarray*=0.001

if __name__ == '__main__':

    vad = Vad()
    vad.set_mode(3)
    sample_dtype = np.int16
    # audio
    fs = 16000
    channels = 8
    blocksize = 320
    fig = go.Figure()
    bytes_per_sample = np.dtype(sample_dtype).itemsize

    bpf = channels * blocksize * bytes_per_sample
    frames =[]
    print("Bytes per packet:", bpf)
    # communication
    IP = "0.0.0.0"
    PORT = 9999
    addr = (IP, PORT)

    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,12),subplot_kw={'projection': 'polar'})

    freqs=np.fft.rfftfreq(2*blocksize,1/fs)         
    theta = np.linspace(-np.pi,np.pi,360)
    look_vec=np.array((np.cos(theta),np.sin(theta),np.zeros(360)))
    steer_delay=micarray@look_vec/343
    exponent=np.exp(-1j*(2*np.pi*np.repeat(np.reshape(freqs[1000*blocksize/fs:9000*blocksize/fs],(1,9000*blocksize/fs-1000*blocksize/fs)),8,axis=0)[:,:,np.newaxis]*steer_delay[:,np.newaxis]))

    #Music 
    r_matrices=[]
    peak_list=[]
    D=1 #number of sources (used in MUSIC)


    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sockd:
        sockd.bind(addr)

        while True:
            block, _ = sockd.recvfrom(bpf)
            if not block:
                break
            while len(block) < bpf:
                foo = sockd.recv(bpf - len(block))
                block = block + foo
            block = np.frombuffer(block, sample_dtype)
            block = block.reshape((channels, blocksize))
            ax[0].cla()
            ax[1].cla()
            if(vad.is_speech(block[0],fs)):
                frames.append(block)

                peak=LPC_freq_estimate(block[0],height=15,order=80,plot=False)
               
                X=np.fft.rfft(block,2*blocksize-1)
                r_matrices.append(X)
                peak_list.append(peak)
            
                if(len(r_matrices)==10):
               
                    averag=np.array(frames)
                    averag=np.mean(averag,axis=0)
                    peak_sum=np.concatenate(peak_list).ravel()
                    peak_sum=np.bincount(peak_sum).argmax()
                    
                    R_Matrix=r_matrix(r_matrices[0],peak_sum)+r_matrix(r_matrices[1],peak_sum)+r_matrix(r_matrices[2],peak_sum)+r_matrix(r_matrices[3],peak_sum)+r_matrix(r_matrices[4],peak_sum)+r_matrix(r_matrices[5],peak_sum)+r_matrix(r_matrices[6],peak_sum)+r_matrix(r_matrices[7],peak_sum)+r_matrix(r_matrices[8],peak_sum)+r_matrix(r_matrices[9],peak_sum)

                    Pmusic=MUSIC(R_Matrix/10,D,micarray,freqs[peak_sum])
                    ax[0].plot(theta,Pmusic,"k")
                    P = srp_phat(averag,exponent,blocksize,fs)   
                    r_matrices.remove(r_matrices[0])
                    peak_list.remove(peak_list[0])   
                    
                    ax[1].plot(np.roll(theta,180),P)
                    ax[1].plot(np.roll(theta,180)[np.argmax(P)],P[np.argmax(P)],'ro')
                    plt.pause(0.0000000001)
            else:
                ax[0].plot(0,0)
                ax[1].plot(0,0)
                plt.pause(0.0000000001)    
