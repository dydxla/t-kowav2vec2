import os
from pathlib import Path
import librosa
import random
from pathlib import Path
import numpy as np
import glob
from typing import Dict, List
from kowav2vec2.finetune.core.datasets.process.utils import find_all_audio_files

def cal_rms(amp):
    # 진폭값의 평균 제곱근(Root Mean Square, RMS) 리턴
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms

def mix_noise(wav):
    clean_amp = wav

    current_file = Path(__file__).resolve()
#     noise_list = glob.glob('./dvc/noise/data/*.wav')
    noise_folder = os.path.join(current_file.parents[5], 'noise')
    noise_list = glob.glob('%s/d*.wav'%noise_folder)+\
                 glob.glob('%s/Rn*.wav'%noise_folder)+\
                 glob.glob('%s/t*.wav'%noise_folder)+\
                 glob.glob('%s/s*.wav'%noise_folder)+\
                 glob.glob('%s/o*.wav'%noise_folder)+\
                 glob.glob('%s/n*.wav'%noise_folder)
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

def get_audio_paths(
        train_root_dir: str = None, 
        eval_root_dir: str = None, 
        format: str = "wav",
        test: bool = False
    ) -> Dict:
    """
    wav 데이터들의 경로를 담은 리스트를 반환하는 메서드.

    Args:
        train_root_dir (str): 불러 오고자 하는 오디오 파일(train)들이 담긴 경로 (default: None)
        eval_root_dir (str): 불러 오고자 하는 오디오 파일(validation)들이 담긴 경로 (default: None)
        format (str): 불러 오고자 하는 오디오 파일의 확장자
        test (bool): test 환경 실행 여부 (default: False)
        
    Returns:
        Dict : 불러 오고자 하는 wav 파일들의 경로들을 담은 리스트
    """
    
    # test 환경인 경우
    # samples/test/audio 폴더로 부터 audio 파일 경로들을 가져옴
    if test:
        current_file = Path(__file__).resolve()
        wav_files = find_all_audio_files(
            root_dir=current_file.parents[5] / "samples" / "audio" / "test",
            format="wav"
        )
        return {'train': wav_files}
    
    # train_root_dir 이 None 인 경우
    # samples/train/audio 폴더로 부터 audio 파일 경로들을 가져옴
    if not train_root_dir:
        current_file = Path(__file__).resolve()
        wav_files = find_all_audio_files(
            root_dir=current_file.parents[5] / "samples" / "audio" / "train",
            format="wav"
        )
        # eval_root_dir 이 None 인 경우
        # samples/validation/audio 폴더로 부터 audio 파일 경로들을 가져옴
        if not eval_root_dir:
            val_wav_files = find_all_audio_files(
                root_dir=current_file.parents[5] / "samples" / "audio" / "validation",
                format="wav"
            )
            return {"train": wav_files, "test": val_wav_files}
        
        val_wav_files = find_all_audio_files(
            root_dir=eval_root_dir,
            format=format
        )
        return {"train": wav_files, "test": val_wav_files}
    
    wav_files = find_all_audio_files(
        root_dir=train_root_dir,
        format=format
    )
    
    if eval_root_dir:
        val_wav_files = find_all_audio_files(
            root_dir=eval_root_dir,
            format=format
        )
        return {"train": wav_files, "test": val_wav_files}
    
    return {"train": wav_files}


def audio_to_signal(file_path, sr: int = 16000):
    signal = librosa.load(file_path, sr=sr)[0]
    return signal
    

def get_audio_signal(file_paths) -> List:
    """
    wav 데이터들의 음성 시그널을 담은 리스트를 반환하는 메서드

    Args:
        file_paths (str or List): 음성 시그널로 변환하고자 하는 오디오 파일 경로 혹은 경로들을 담은 리스트
        
    Returns:
        List : 음성 시그널을 담은 리스트
    """
    if isinstance(file_paths, str):
        return [audio_to_signal(file_paths)]
    elif isinstance(file_paths, List):
        signals = [audio_to_signal(path) for path in file_paths]
        return signals
    else:
        raise ValueError("file_paths must be a string or a list of strings")