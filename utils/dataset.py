# Pytorch Import
import torch

# Data Process Library Import
import librosa
import random
from collections import defaultdict
from utils import mix_noise

class CustomDataset(torch.utils.data.Dataset):
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
    def __init__(self, data, target, processor, training=True, samples_per_class=50):
        """
        params:
            data: 데이터 경로 리스트
            target: 데이터 라벨 리스트
            processor: 데이터 처리할 프로세서
            training: 학습데이터인지 평가데이터인지
            samples_per_class: 다운샘플링 하고자 하는 데이터 개수
        """
        self.data = data
        self.target = target
        self.processor = processor
        self.mix_noise = mix_noise
        self.training = training
        self.samples_per_class = samples_per_class
        self.class_data = self._organize_data_by_class()
        self.current_epoch_data = self._sample_data()
        # print("current_epoch_data length : ",len(self.current_epoch_data))
        # print("current_epoch_data 1 : ", self.current_epoch_data[0])


    def _organize_data_by_class(self):
        class_data = defaultdict(list)
        for path, label in zip(self.data, self.target):
            class_data[label].append(path)
        return class_data
        

    def _sample_data(self,):
        sampled_data = []
        sampled_target = []
        for label, paths in self.class_data.items():
            if len(paths) > self.samples_per_class:
                sampled_paths = random.sample(paths, self.samples_per_class)
            else:
                sampled_paths = paths
            sampled_data.extend(sampled_paths)
            sampled_target.extend([label] * len(sampled_paths))
        return list(zip(sampled_data, sampled_target))
            

    def __len__(self):
        return len(self.current_epoch_data)

    def __getitem__(self, idx):
        # path = self.data[idx]
        # label = self.target[idx]
        # print("idx : ",idx)
        path, label = self.current_epoch_data[idx]
        speech = librosa.load(path, sr=16000)[0]
        if self.training:
            if random.random() > 0.5:
                speech = self.mix_noise(speech)
        processed = self.processor(speech, sampling_rate=16000, return_tensors='pt', padding='longest', text=label)
        processed["input_length"] = len(processed["input_values"][0])
        dataset = {"input_values":[processed["input_values"][0]],
                "attention_mask":[processed["attention_mask"][0]],
                "labels":[processed["labels"][0]]}
        return dataset

    def on_epoch_end(self):
        self.current_epoch_data = self._sample_data()