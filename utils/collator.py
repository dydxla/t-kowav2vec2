from transformers import Wav2Vec2Processor
from typing import Union, Dict
import torch
from dataclasses import dataclass, field

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"

    def __call__(self, features) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"][0]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        
        # del input_features
        # del label_features
        # del labels_batch
        # del labels
        # gc.collect()
        # torch.cuda.empty_cache()

        return batch