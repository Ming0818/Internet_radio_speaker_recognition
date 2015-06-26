__author__ = 'Radek W'

"""
#######################################################################################################
If you use this project in scientific publication, we would appreciate citations to the following paper:
@article{WeychanAsrPi,
    author = {R. Weychan and T. Marciniak and  A. Dabrowski},
    title = {Implementation aspects of speaker recognition using {Python} language and {Raspberry Pi} platform},
    journal = {IEEE SPA: Signal Processing Algorithms, Architectures, Arrangements, and Applications Conference Proceedings},
    year = {2015},
    pages = {95--98},
    confidential = {n}
}

#######################################################################################################
"""

import urllib
#import wave
import string
import ctypes
import binascii
import time

import pymedia.audio.acodec as acodec
import pymedia.muxer as muxer
import pymedia.audio.sound as sound
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import mixture
import scipy.io as sio
#import scipy.io.wavfile as wav
import pygame
#from scikits.audiolab import play

import MFCC


# import pyglet
# import pyaudio


def read_radio_stream(url_):

    database = sio.loadmat('mfcc_16_fft256_GMM.mat')
    database.pop('__header__')
    database.pop('__version__')
    database.pop('__globals__')

    r2 = urllib.urlopen(url_)
    pygame.mixer.init(44100, -16, 2, 2048)
    print pygame.mixer.get_init()
    chan1 = pygame.mixer.find_channel()

    format = sound.AFMT_S16_LE
    print sound.getODevices()
    #snd_out = sound.Output(44100, 2, format)

    dm = muxer.Demuxer('mp3')
    dec = None
    snd = None

    print(r2.info())
    print('###################\n')

    #f = open('radio.mp3', 'wb')
    #g = open('radio.wav', 'wb')
    i = 0
    while True:  #i < 3:

        samples = r2.read(15000)

        frames = dm.parse(samples)

        if dec is None:
            # Open decoder
            dec = acodec.Decoder(dm.streams[0])
        

        #start = time.time()
        sound_np_array = ansic_to_numpy(frames, dec)
        #print (sound_np_array.shape[0])/44100.0
        #elapsed = (time.time() - start)
        #print 'decode and ndaray - %2.8f' %elapsed
        
        #start = time.time()
        to_play = np.array(np.repeat(sound_np_array[:, np.newaxis], 2, 1), dtype = 'int16')
        sounds = pygame.sndarray.make_sound(to_play)
        chan1.queue(sounds)
        #elapsed = (time.time() - start)
        #print 'to play - %2.8f' %elapsed

        #start = time.time()
        sound_np_array = decimate(sound_np_array, 4)
        #elapsed = (time.time() - start)
        #print 'downsample - %2.8f' %elapsed

        #start = time.time()
        mfcc_features = MFCC.extract(sound_np_array) #1.5s
        mfcc_features = mfcc_features[:, 1:]
        #elapsed = (time.time() - start)
        #print 'mfcc - %2.8f' %elapsed


        g = mixture.GMM(n_components=16)
        log_prob = -10000
        winner = 'nobody'

        for key, values in database.iteritems():
            try:
                g.means_ = values[0, :, :]
                g.covars_ = values[1, :, :]
                g.weights_ = values[2, :, 1]
                
                #start = time.time()
                temp_prob = np.mean(g.score(mfcc_features))
                #elapsed = (time.time() - start)
                #print 'log-likelihood - %2.8f' %elapsed
                
                if temp_prob > log_prob:
                    log_prob = temp_prob
                    winner = key
            except TypeError:
                print 'error dla ', key

        print winner, log_prob

    print('\n###################')


def decimate(np_array, factor):
    np_array = np_array[0:(np_array.shape[0] / factor) * factor]
    np_array = np_array.reshape(-1, factor)
    np_array = np_array.reshape(-1, factor).mean(axis=1)
    return np_array


def calc_rms(np_array):
    return np.sqrt(sum(np_array ** 2))


def plot_audio_data(np_array):
    plt.plot(np_array)
    plt.show()


def ansic_to_numpy_old(frames_, dec_):
    scale_fun = lambda x: x / 1.0 if x < 32768 else (x - 65536) / 1.0  # uint16 to int16
    sound_np_array_ = np.array([])

    for fr in frames_:
        r = dec_.decode(fr[1])
        if r and r.data:
            # snd_out.play(r.data)
            #raw_ansic_python_obj = ctypes.py_object(r.data)
            #ACstr_raw_C_data = raw_ansic_python_obj.value
            #hex_values_str = ACstr_raw_C_data.__str__()
            hex_values_str = r.data.__str__()
            #hex_audio_mono1 = hex_values_str[0:-1:2]
            hex_audio_mono = ''.join([hex_values_str[i:i + 2] for i in range(0, len(hex_values_str), 4)])
            #pcm_audio_uint16 = [int(binascii.b2a_hex(hex_audio_mono[i:i+2]), 16) for i in range(1, len(hex_audio_mono), 2)] #bigendian
            pcm_audio_uint16 = [int(binascii.b2a_hex(hex_audio_mono[i:i - 2:-1]), 16) for i in
                                range(3, len(hex_audio_mono), 2)]  #little endian
            pcm_audio = np.array([scale_fun(x) for x in pcm_audio_uint16])
            sound_np_array_ = np.append(sound_np_array_, pcm_audio, axis=1)
    return sound_np_array_
    

def ansic_to_numpy(frames_):
    sound_np_array_ = np.array([])

    for fr in frames_:
        hex_values_str = fr.__str__()
        pcm_audio = np.fromstring(hex_values_str, dtype='int16')
        pcm_audio = pcm_audio[2::2]
        sound_np_array_ = np.append(sound_np_array_, pcm_audio, axis=1)
    return sound_np_array_


if __name__ == "__main__":
    url = 'http://poznan5-4.radio.pionier.net.pl:8000/tuba10-1.mp3'
    read_radio_stream(url)


