import os
from pathlib import Path
from typing import List

def find_all_audio_files(root_dir: str, format: str = "wav") -> List:
    """
    root 경로의 하위 경로들에 있는 모든 오디오 파일 절대경로들을 리스트에 담아 반환하는 메서드
    
    Args:
        root_dir (str): 데이터가 있는 root 경로
        format (str): 찾고자 하는 오디오 파일의 확장자

    Returns:
        List : 오디오 파일 절대경로들을 담은 리스트
    """
    if not Path(root_dir).exists():
        raise ValueError("root_dir is not exists.")
    
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(f".{format}"):
                audio_files.append(os.path.join(dirpath, filename))
    return audio_files


def find_all_label_files(root_dir: str, format: str = "txt") -> List:
    """
    root 경로의 하위 경로들에 있는 모든 label 파일 절대경로들을 리스트에 담아 반환하는 메서드
    
    Args:
        root_dir (str): 데이터가 있는 root 경로
        format (str): 찾고자 하는 label 파일의 확장자 (default: 'txt')

    Returns:
        List : label 파일 절대경로들을 담은 리스트
    """
    if not Path(root_dir).exists():
        raise ValueError("root_dir is not exists.")
    
    label_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(f".{format}"):
                label_files.append(os.path.join(dirpath, filename))
    return label_files

