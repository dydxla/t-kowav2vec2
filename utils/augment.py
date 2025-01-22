import librosa
import numpy as np
import random
import os
import glob

def cal_rms(amp):
    # 진폭값의 평균 제곱근(Root Mean Square, RMS) 리턴
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms

def mix_noise(wav):
    clean_amp = wav
#     noise_list = glob.glob('./dvc/noise/data/*.wav')
    src_folder = os.path.join(os.path.dirname(__file__), '..', 'src')
    noise_list = glob.glob('%s/noise/d*.wav'%src_folder)+\
                 glob.glob('%s/noise/Rn*.wav'%src_folder)+\
                 glob.glob('%s/noise/t*.wav'%src_folder)+\
                 glob.glob('%s/noise/s*.wav'%src_folder)+\
                 glob.glob('%s/noise/o*.wav'%src_folder)+\
                 glob.glob('%s/noise/n*.wav'%src_folder)
    noise_randidx = random.randint(0, len(noise_list)-1)
    noise_amp = librosa.load(noise_list[noise_randidx], sr=16000)[0]

    start = random.randint(0, len(noise_amp)-len(clean_amp))
    clean_rms = cal_rms(clean_amp)
    split_noise_amp = noise_amp[start:start+len(clean_amp)]
    noise_rms = cal_rms(split_noise_amp)

#     snr = random.uniform(0, 15)
    snr = random.choice([0, 5, 10, 15])
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)

    adjusted_noise_amp = split_noise_amp*(adjusted_noise_rms/noise_rms)
    mixed_amp = (clean_amp + adjusted_noise_amp)

    if (mixed_amp.max(axis=0) > 1):
        mixed_amp = mixed_amp * (1/mixed_amp.max(axis=0))
        clean_amp = clean_amp * (1/mixed_amp.max(axis=0))
        adjusted_noise_amp = adjusted_noise_amp * (1/mixed_amp.max(axis=0))
    return mixed_amp