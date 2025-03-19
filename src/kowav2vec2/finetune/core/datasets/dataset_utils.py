import pickle
import glob
import os
import torch
import librosa
import random
from collections import defaultdict
from pathlib import Path
from kowav2vec2.finetune.configs import _conf
from kowav2vec2.finetune.core.datasets.process import \
    (get_audio_paths,
     get_label_paths,
     get_audio_signal,
     mix_noise)
from kowav2vec2.finetune.core.datasets.process.utils import path_to_basename

def load_dataset(
        train_audio_root_dir: str = None, 
        eval_audio_root_dir: str = None,
        train_label_root_dir: str = None, 
        eval_label_root_dir: str = None,
        audio_format: str = "wav",
        label_format: str = "txt",
        test: bool = False
    ):
    """
    오디오 데이터 경로와 label 을 불러오는 메서드.

    Args:
        train_root_dir (str): 학습 데이터의 루트 경로 (default: None)
        eval_root_dir (str): 평가 데이터의 루트 경로 (default: None)
        audio_format (str): 불러 오고자 하는 오디오 파일의 확장자 (default: 'wav')
        label_format (str): 불러 오고자 하는 label 파일의 확장자 (default: 'txt')
        test (bool): test 환경 여부

    Returns:
        List, List : 오디오 경로 리스트, label 리스트
    """
    try:
        # if test:
        #     get_audio_paths(test=True)
        # 테스트 환경에 따른 로직 추가하기
            
        # 1. 오디오 파일 경로 가져오기
        # 2. 오디오파일명과 매칭되는 텍스트파일(Label) 가져오기
        #   2-0. 먼저 텍스트파일 경로들을 모두 가져오기
        #   2-1. 텍스트파일경로에서 오디오파일명과 매칭되는 경로들로 재가공
        # 3. 오디오파일들을 시그널로 처리
        # 4. 처리된 오디오, 텍스트 반환

        # 오디오 파일 경로 가져오기
        audio_dataset = get_audio_paths(    # Dict
            train_root_dir=train_audio_root_dir,
            eval_root_dir=eval_audio_root_dir,
            format=audio_format,
            test=test
        )

        label_dataset = get_label_paths(    # Dict
            train_root_dir=train_label_root_dir,
            eval_root_dir=eval_label_root_dir,
            format=label_format,
            test=test
        )

        label_dict = {Path(p).stem: p for p in label_dataset['train']}
        label_dataset['train'] = [label_dict[Path(audio).stem] for audio in audio_dataset['train']]
        if label_dataset['test'] and audio_dataset['test']:
            label_dataset['test'] = [label_dict[Path(audio).stem] for audio in audio_dataset['test']]
        
        audio_dataset['train'] = get_audio_signal(audio_dataset['train'])
        if audio_dataset['test']:
            audio_dataset['test'] = get_audio_signal(audio_dataset['test'])
        
        return audio_dataset, label_dataset

    except Exception as e:
        print(e)
        return {'info': f'<Error message> : {e} '}


class KoWav2Vec2Dataset(torch.uitls.data.Dataset):
    """
    custom dataset class
    functions:
        __init__: 인스턴스 초기화
        _organize_data_by_class: 각 라벨의 데이터를 라벨별로 딕셔너리 저장 및 정렬된 딕셔너리 반환
        _sample_data: 해당 라벨의 데이터가 samples_per_class 보다 많은 경우 랜덤으로 samples_per_class 수만큼의 데이터 추출
        __len__: 한번의 epoch에 사용될 데이터 총개수
        __getitem__: 데이터 처리 및 처리된 데이터 반환
        on_epoch_end: 한 에폭의 학습이 끝난뒤 실행되는 함수(_sample_data 를 다시 호출하여 데이터 랜덤으로 재세팅)
    """
    def __init__(
            self, 
            audio_data, 
            label_data, 
            processor, 
            training=True,
        ):
        """
        params:
            data: 데이터 경로 리스트
            target: 데이터 라벨 리스트
            processor: 데이터 처리할 프로세서
            training: 학습데이터인지 평가데이터인지
            samples_per_class: 다운샘플링 하고자 하는 데이터 개수
        """
        self.audio_data = audio_data
        self.label_data = label_data
        self.processor = processor
        self.mix_noise = mix_noise
        self.training = training
        self.current_epoch_data = self._sample_data()


    def _sample_data(self,):
        """
        audio data, label data를 zip으로 묶어 list로 반환하는 매서드
        매서드 호출때 마다 순서가 무작위로 섞임
        """
        paired = list(zip(self.audio_data, self.label_data))
        random.shuffle(paired)
        return paired
            

    def __len__(self):
        return len(self.current_epoch_data)


    def __getitem__(self, idx):
        # path = self.data[idx]
        # label = self.target[idx]
        # print("idx : ",idx)
        audio, label = self.current_epoch_data[idx]
        if self.training:
            if random.random() > _conf["mix_noise_ratio"]:
                audio = self.mix_noise(audio)
        processed = self.processor(audio, sampling_rate=_conf["sampling_rate"], return_tensors='pt', padding='longest', text=label)
        processed["input_length"] = len(processed["input_values"][0])
        dataset = {"input_values":[processed["input_values"][0]],
                "attention_mask":[processed["attention_mask"][0]],
                "labels":[processed["labels"][0]]}
        return dataset


    def on_epoch_end(self):
        self.current_epoch_data = self._sample_data()