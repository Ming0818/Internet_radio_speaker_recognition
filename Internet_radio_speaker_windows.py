__author__ = 'Radek W'

import urllib
import wave
import string
import ctypes
import binascii
import time

import pymedia.audio.acodec as acodec
import pymedia.muxer as muxer
import pymedia.audio.sound as sound
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import scipy.io as sio
import scipy.io.wavfile as wav

import MFCC

def read_radio_stream(url_):

    database = sio.loadmat('mfcc_16_fft256_GMM.mat')
    database.pop('__header__')
    database.pop('__version__')
    database.pop('__globals__')

    r2 = urllib.urlopen(url_)

    format_ = sound.AFMT_S16_LE
    snd_out = sound.Output(44100, 2, format_)

    dm = muxer.Demuxer('mp3')
    dec = None
    snd = None

    print(r2.info())
    print('###################\n')


    while True:

        samples = r2.read(15000)

        frames = dm.parse(samples)

        if dec is None:
            dec = acodec.Decoder(dm.streams[0])

        #start = time.clock()
        frames_list = decode_frames(frames, dec)
        #elapsed = (time.clock() - start)
        #print "Decode - ", elapsed

        #start = time.clock()
        play_decoded_frames(frames_list, snd_out)
        #elapsed = (time.clock() - start)
        #print "Send to play play - ", elapsed

        #start = time.clock()
        sound_np_array = ansic_to_numpy(frames_list)
        #elapsed = (time.clock() - start)
        #print "To ndarray - ", elapsed

        #start = time.clock()
        sound_np_array = decimate(sound_np_array, 2)
        #elapsed = (time.clock() - start)
        #print "Decimate - ", elapsed

        #start = time.clock()
        mfcc_features = MFCC.extract(sound_np_array) #1.5s
        mfcc_features = mfcc_features[:, 1:]
        #elapsed = (time.clock() - start)
        #print "MFCC - ", elapsed

        #print mfcc_features.shape

        g = mixture.GMM(n_components=16)
        log_prob = -10000
        winner = 'nobody'

        for key, values in database.iteritems():
            try:
                #start = time.clock()
                g.means_ = values[0, :, :]
                g.covars_ = values[1, :, :]
                g.weights_ = values[2, :, 1]
                temp_prob = np.mean(g.score(mfcc_features))
                if temp_prob > log_prob:
                    log_prob = temp_prob
                    winner = key
                #elapsed = (time.clock() - start)
                #print "Log-likelihood - ", elapsed
            except TypeError:
                print 'error for ', key

        print winner, log_prob

    print('\n###################')


def decode_frames(frames_, dec_):
    frames_list_ = []
    for fr in frames_:
        r = dec_.decode(fr[1])
        frames_list_.append(r.data)
    return frames_list_


def play_decoded_frames(frames_, snd_out_):
    for fr in frames_:
        snd_out_.play(fr)
    #snd_out_.play(frames_)


def ansic_to_numpy_old(frames_):
    scale_fun = lambda x: x / 1.0 if x < 32768 else (x - 65536) / 1.0  # uint16 to int16
    sound_np_array_ = np.array([])

    for fr in frames_:
        hex_values_str = fr.__str__()
        hex_audio_mono = ''.join([hex_values_str[i:i + 2] for i in range(0, len(hex_values_str), 4)])
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



if __name__ == "__main__":
    url = 'http://poznan5-4.radio.pionier.net.pl:8000/tuba10-1.mp3'
    read_radio_stream(url)

