__author__ = 'Radoslaw Weychan'


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


# All TIMIT files should be placed in speakers directory. 
# In this case the name of every file is like XXXyy.wav, 
# where XXXX stands for person 'name' (like FADG from TIMIT files)
# and yy stands for number of utterrance (00 to 09)
# FADG00.wav, FADG01.wav, FADG02.wav

import glob
import os
import numpy as np
import MFCC
import scipy.io.wavfile as wav
from sklearn import mixture
import scipy.io as sio


def GMM_test(ii):
    speakers_MFCC_dict = {}
    speaker_GMM_dict = {}
    files = glob.glob(os.getcwd()+'\\speakers\\*.wav')
    gauss_num = 32
    iterator = 1
    num_iter = ii


    if os.path.isfile('mfcc_'+str(gauss_num)+'.mat'):
        speaker_GMM_dict = sio.loadmat('mfcc_'+str(gauss_num)+'.mat')
        speaker_GMM_dict.pop('__header__')
        speaker_GMM_dict.pop('__version__')
        speaker_GMM_dict.pop('__globals__')
    else:
        for file in files:
            #print(file)
            if file[-6:-4] == '00':   #file[len(file)-12:len(file)-9]
                current_speaker = file[len(file)-10:len(file)-6]
                print("############# Calculate MFCC and GMM for ", current_speaker, " , speaker no ", str(iterator))
                #if iterator == 572:
                #    print("Tu bedzie error")

                iterator += 1
                merged_files = np.array([])
                for i in range(0, 9):
                    current_file = wav.read(file[:-5]+str(i)+file[-4:])
                    merged_files = np.append(merged_files, current_file[1])
                #print(type(merged_files))
                speaker_MFCC = MFCC.extract(merged_files)
                speaker_MFCC = speaker_MFCC[:, 1:]

                speakers_MFCC_dict[current_speaker] = speaker_MFCC
                g = mixture.GMM(n_components=gauss_num, n_iter=num_iter)
                g.fit(speaker_MFCC)

                speaker_model = np.array([g.means_, g.covars_, np.repeat(g.weights_[:, np.newaxis], 12, 1)])
                speaker_GMM_dict[current_speaker] = speaker_model


        sio.savemat('mfcc_'+str(gauss_num)+'.mat', speaker_GMM_dict, oned_as='row')


    iterator = 1
    good = 0
    bad = 0
    total = 0

    for file in files:
        if file[-6:-4] == '09':
            g = mixture.GMM(n_components=gauss_num, n_iter=num_iter)
            current_file = wav.read(file)
            current_speaker = file[len(file)-10:len(file)-6]
            #print(current_speaker, )
            speaker_MFCC = MFCC.extract(current_file[1])
            speaker_MFCC = speaker_MFCC[:, 1:]
            log_prob = -10000
            winner = 'nobody'
            for key, values in speaker_GMM_dict.items():
                try:
                    g.means_ = values[0, :, :]
                    g.covars_ = values[1, :, :]
                    g.weights_ = values[2, :, 1]
                    temp_prob = np.mean(g.score(speaker_MFCC))
                    if temp_prob > log_prob:
                        log_prob = temp_prob
                        winner = key
                except TypeError:
                    print('error for ', key)
            if current_speaker == winner:
                good += 1
            else:
                bad += 1
            total +=1
            print(current_speaker, " speaker no ", str(iterator), " is similar to ", winner, " - log prob = ", str(log_prob))
            print("good = ", str(good), ", bad = ", str(bad), ", total = ", str(total))
            iterator += 1

    print("GMM, n_iter = ", num_iter, ", Efficiency = ", str(good/total))




def VBGMM_test(cov_type, alpha_val):
    #speakers_MFCC_dict = {}
    #speaker_GMM_dict = {}
    files = glob.glob(os.getcwd()+'\\speakers\\*.wav')
    gauss_num = 32
    iterator = 1
    test_files = []
    good = 0
    bad = 0
    total = 0

    for file in files:
        if file[-6:-4] == '09':
            test_files.append(file)

    for file in files:
        #print(file)
        if file[-6:-4] == '00':   #file[len(file)-12:len(file)-9]
            current_speaker = file[len(file)-10:len(file)-6]
            #print("############# Calculate MFCC and VBGMM for ", current_speaker, " , speaker no ", str(iterator))

            merged_files = np.array([])
            for i in range(0, 9):
                current_file = wav.read(file[:-5]+str(i)+file[-4:])
                merged_files = np.append(merged_files, current_file[1])
            #print(type(merged_files))
            speaker_MFCC = MFCC.extract(merged_files)
            speaker_MFCC = speaker_MFCC[:, 1:]
            #speakers_MFCC_dict[current_speaker] = speaker_MFCC
            g = mixture.VBGMM(n_components=gauss_num, n_iter=100, covariance_type=cov_type, alpha=alpha_val)
            g.fit(speaker_MFCC)
            #speaker_model = np.array([g.means_, g.precs_, np.repeat(g.weights_[:, np.newaxis], 12, 1)])
            #speaker_GMM_dict[current_speaker] = speaker_model
            log_prob = -10000
            winner = 'nobody'
            for test_file in test_files:
                current_test_speaker = test_file[len(test_file)-10:len(test_file)-6]
                current_test_file = wav.read(test_file)
                test_speaker_MFCC = MFCC.extract(current_test_file[1])
                test_speaker_MFCC = test_speaker_MFCC[:, 1:]
                temp_prob = np.mean(g.score(test_speaker_MFCC))
                if temp_prob > log_prob:
                    log_prob = temp_prob
                    winner = current_test_speaker
            if winner == current_speaker:
                good += 1
            else:
                bad += 1
            total +=1
            #print(current_speaker, " speaker no ", str(iterator), " is similar to ", winner, " - log prob = ", str(log_prob))
            #print("good = ", str(good), ", bad = ", str(bad), ", total = ", str(total))
            iterator += 1

    print("VBGMM (covariance_type - ", cov_type, ", alpha - ", str(alpha_val), "), Efficiency = ", str(good/total))
    #sio.savemat('mfcc_'+str(gauss_num)+'_VBGMM.mat', speaker_GMM_dict, oned_as='row')



def DPGMM_test(cov_type, alpha_val):
    #speakers_MFCC_dict = {}
    #speaker_GMM_dict = {}
    files = glob.glob(os.getcwd()+'\\speakers\\*.wav')
    gauss_num = 32
    iterator = 1
    test_files = []
    good = 0
    bad = 0
    total = 0

    for file in files:
        if file[-6:-4] == '09':
            test_files.append(file)

    for file in files:
        #print(file)
        if file[-6:-4] == '00':   #file[len(file)-12:len(file)-9]
            current_speaker = file[len(file)-10:len(file)-6]
            #print("############# Calculate MFCC and DPGMM for ", current_speaker, " , speaker no ", str(iterator))
            #if iterator == 572:
            #    print("Tu bedzie error")

            merged_files = np.array([])
            for i in range(0, 9):
                current_file = wav.read(file[:-5]+str(i)+file[-4:])
                merged_files = np.append(merged_files, current_file[1])
            #print(type(merged_files))
            speaker_MFCC = MFCC.extract(merged_files)
            speaker_MFCC = speaker_MFCC[:, 1:]
            #speakers_MFCC_dict[current_speaker] = speaker_MFCC
            g = mixture.DPGMM(n_components=gauss_num, n_iter=100, covariance_type=cov_type, alpha=alpha_val)
            g.fit(speaker_MFCC)
            #speaker_model = np.array([g.means_, g.precs_, np.repeat(g.weights_[:, np.newaxis], 12, 1)])
            #speaker_GMM_dict[current_speaker] = speaker_model
            log_prob = -10000
            winner = 'nobody'
            for test_file in test_files:
                current_test_speaker = test_file[len(test_file)-10:len(test_file)-6]
                current_test_file = wav.read(test_file)
                test_speaker_MFCC = MFCC.extract(current_test_file[1])
                test_speaker_MFCC = test_speaker_MFCC[:, 1:]
                temp_prob = np.mean(g.score(test_speaker_MFCC))
                if temp_prob > log_prob:
                    log_prob = temp_prob
                    winner = current_test_speaker
            if winner == current_speaker:
                good += 1
            else:
                bad += 1
            total +=1
            #print(current_speaker, " speaker no ", str(iterator), " is similar to ", winner, " - log prob = ", str(log_prob))
            #print("good = ", str(good), ", bad = ", str(bad), ", total = ", str(total))
            iterator += 1

    print("DPGMM (covariance_type - ", cov_type, ", alpha - ", str(alpha_val), "), Efficiency = ", str(good/total))
    #sio.savemat('mfcc_'+str(gauss_num)+'_VBGMM.mat', speaker_GMM_dict, oned_as='row')



if __name__ == "__main__":

    GMM_test(10)



    #alpha_num = [1, 10, 100]
    #covariance_type = ["spherical", "tied", "diag", "full"]

    #for i in alpha_num:
    #    for j in covariance_type:
    #       DPGMM_test(j, i)
    #       VBGMM_test(j, i)