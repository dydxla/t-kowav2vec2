import os
import librosa
import glob
from typing import List

def get_audio_paths(root_dir: str = None, test: bool = False) -> List:
    """
    wav 데이터들의 경로를 담은 리스트를 반환하는 메서드.

    Args:
        root_dir (str): 불러 오고자 하는 wav 파일들이 담긴 경로 (default: None)
        test (bool): test 환경 실행 여부 (default: False)
        
    Returns:
        List : 불러 오고자 하는 wav 파일들의 경로들을 담은 리스트
    """
    def fine_all_wav_files(root_path):
        wav_files = []
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.endswith(".wav"):
                    wav_files.append(os.path.join(dirpath, filename))
        return wav_files
    
    if test:
        wav_files = fine_all_wav_files("./src/test/audio")
        return wav_files
    if not root_dir:
        raise ValueError("root_dir param has no value.")
    wav_files = fine_all_wav_files(root_dir)
    return wav_files
    

def get_audio_signal(file_paths) -> List:
    """
    wav 데이터들의 음성 시그널을 담은 리스트를 반환하는 메서드

    Args:
        file_paths (str or List): 음성 시그널로 변환하고자 하는 오디오 파일 경로 혹은 경로들을 담은 리스트
        
    Returns:
        List : 음성 시그널을 담은 리스트
    """
    def load_signal(file_path):
        signal = librosa.load(file_path, sr=16000)[0]
        return signal
    if isinstance(file_paths, str):
        return load_signal(file_paths)
    elif isinstance(file_paths, List):
        signals = [load_signal(path) for path in file_paths]
        return signals
    else:
        raise ValueError("file_paths must be a string or a list of strings")