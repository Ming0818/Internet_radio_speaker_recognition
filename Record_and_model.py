__author__ = 'Radek W'


__author__ = 'Radek W'

import urllib
import ctypes
import binascii
import time
import pymedia.audio.acodec as acodec
import pymedia.muxer as muxer
import pymedia.audio.sound as sound
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import scipy.io.wavfile as wav
import scipy.io as sio
import os
import MFCC


def read_radio_stream(url_):

    an_array = np.array([])
    r2 = urllib.urlopen(url_)

    format_ = sound.AFMT_S16_LE
    snd_out = sound.Output(44100, 2, format_)

    dm = muxer.Demuxer('mp3')
    dec = None

    print(r2.info())
    print('###################\n')

    i = 0

    while i < 1:  # in the case of longer signals needed

        start = time.time()

        samples = r2.read(300000)

        frames = dm.parse(samples)

        if dec is None:
            # Open decoder
            dec = acodec.Decoder(dm.streams[0])

        play_frames(frames, dec, snd_out)

        sound_np_array = ansic_to_numpy(frames, dec)

        sound_np_array = decimate(sound_np_array, 2)
        an_array = np.append(an_array, np.transpose(sound_np_array))

        end = time.time()
        print end-start
        i += 1
    return an_array


def play_frames(frames_, dec_, snd_out_):
    for fr in frames_:
        r = dec_.decode(fr[1])
        if r and r.data:
            snd_out_.play(r.data)


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


def ansic_to_numpy(frames_, dec_):
    scale_fun = lambda x: x / 1.0 if x < 32768 else (x - 65536) / 1.0  # uint16 to int16
    sound_np_array_ = np.array([])

    for fr in frames_:
        r = dec_.decode(fr[1])
        if r and r.data:
            # raw_ansic_python_obj = ctypes.py_object(r.data)
            # ACstr_raw_C_data = raw_ansic_python_obj.value
            # hex_values_str = ACstr_raw_C_data.__str__()
            hex_values_str = r.data.__str__()
            # hex_audio_mono1 = hex_values_str[0:-1:2]
            hex_audio_mono = ''.join([hex_values_str[i:i + 2] for i in range(0, len(hex_values_str), 4)])
            pcm_audio_uint16 = [int(binascii.b2a_hex(hex_audio_mono[i:i - 2:-1]), 16) for i in
                                range(3, len(hex_audio_mono), 2)]  # little endian
            pcm_audio = np.array([scale_fun(x) for x in pcm_audio_uint16])
            sound_np_array_ = np.append(sound_np_array_, pcm_audio, axis=1)
    return sound_np_array_


def add_to_database(url_, person_name_):
    gmm_models = {}

    if os.path.isfile('mfcc.mat'):
        gmm_models = sio.loadmat('mfcc.mat')
    print "Recording and processing...\n\n"
    full_sound_model = read_radio_stream(url_)

    wav.write('People\\'+person_name_+'.wav', 22050, full_sound_model/32767.0)

    print "Calculating MFCC and saving the model..."
    mfcc_features = MFCC.extract(full_sound_model)
    mfcc_features = mfcc_features[:, 1:]

    g = mixture.GMM(n_components=128)
    g.fit(mfcc_features)
    model = np.array([g.means_, g.covars_, np.repeat(g.weights_[:, np.newaxis], 12, 1)])  # weights have to be repeated to properly save the np array



    print len(g.means_)

    gmm_models[person_name_] = model
    sio.savemat('mfcc.mat', gmm_models, oned_as='row')


if __name__ == "__main__":
    url = 'http://poznan5-4.radio.pionier.net.pl:8000/tuba10-1.mp3'
    person_name = 'Person_1'

    add_to_database(url, person_name)



