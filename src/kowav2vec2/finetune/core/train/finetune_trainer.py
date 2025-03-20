import os
import torch, platform, inspect
from transformers import TrainingArguments
from trl import SFTTrainer
from kowav2vec2.finetune.core.models.utils import model_preprocess
from kowav2vec2.finetune.configs import _train_args, _train_conf, _conf
from kowav2vec2.finetune.core.train.base_trainer import BaseTrainer
from kowav2vec2.finetune.core.train.utils import DataCollatorCTCWithPadding, compute_metrics
from kowav2vec2.finetune.core.models import load_wav2vec2
from kowav2vec2.finetune.core.processor import load_tokenizer_and_processor
from kowav2vec2.finetune.core.datasets import load_dataset, KoWav2Vec2Dataset
# from kowav2vec2.finetune.utils import get_peft_config


class FinetuneTrainer(BaseTrainer):
    def __init__(
            self, 
            model_dir: str = "kresnik/wav2vec2-large-xlsr-korean", 
            is_cache: bool = False,
            tokenizer_dir: str = None,
            dataset_dir: str = "./datasets", 
            model_dtype = torch.float16, 
            initial_configs = None,
            test: bool = False,
            **kwargs
    ):
        super().__init__(initial_configs)
        self.model_dir = model_dir
        self.tokenizer_dir = tokenizer_dir
        self.dataset_dir = dataset_dir

        # Load model and tokenizer
        self.tokenizer, self.processor = load_tokenizer_and_processor(self.model_dir, self.tokenizer_dir)
        self.model = load_wav2vec2(self.model_dir)
        if not is_cache:
            self.model = model_preprocess(self.model, self.tokenizer)
        if _conf["half_model"]:
            self.model = self.model.half()

        # custom dataset directory
        train_audio_root_dir = kwargs.get("train_audio_root_dir", os.path.join(self.dataset_dir, "audio", "train"))
        eval_audio_root_dir = kwargs.get("eval_audio_root_dir", os.path.join(self.dataset_dir, "audio", "validation"))
        train_label_root_dir = kwargs.get("train_label_root_dir", os.path.join(self.dataset_dir, "label", "train"))
        eval_label_root_dir = kwargs.get("eval_label_root_dir", os.path.join(self.dataset_dir, "label", "validation"))

        # Load dataset
        self.audio_dataset, self.label_dataset = load_dataset(
            train_audio_root_dir=train_audio_root_dir,
            eval_audio_root_dir=eval_audio_root_dir if os.path.isdir(eval_audio_root_dir) else None,
            train_label_root_dir=train_label_root_dir,
            eval_label_root_dir=eval_label_root_dir if os.path.isdir(eval_label_root_dir) else None,
            audio_format=kwargs.get("audio_format", _conf["audio_format"]),
            label_format=kwargs.get("label_format", _conf["label_format"]),
            test=test
            )

        # Initialize TrainingArguments
    def __create_training_args(
            self, 
            **kwargs
    ):
        return TrainingArguments(**kwargs)

    def __create_trainer(
            self,
            model, 
            tokenizer, 
            train_dataset, 
            test_dataset, 
            training_args,
            data_collator,
            **kwargs
    ):
        default_args = dict(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        default_args.update(kwargs)
        return SFTTrainer(**default_args)

    def run_finetune(
            self,
            args: dict = None,
            **kwargs
    ):
        """
        run finetuning method

        Args:
            method (str): lora or mora or None
            args (dict): training arguments
            lora_args (dict): lora configs (when. method=="lora")

        Returns:
            
        """

        # 기본 arguments 정의
        if not args:
            args = self.configs
        if platform.system()=="Windows":
            args["deepspeed"] = None

        # 모든 파라미터를 자동으로 가져오기
        training_args_keys = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
        sft_trainer_keys = set(inspect.signature(SFTTrainer.__init__).parameters.keys())

        # 불필요한 키 제거
        training_args_keys -= {'self', 'args', 'kwargs'}
        sft_trainer_keys -= {'self', 'args', 'kwargs'}

        # kwargs에서 분리
        training_args_kwargs = {k: v for k, v in args.items() if k in training_args_keys}
        sft_trainer_kwargs = {k: v for k, v in args.items() if k in sft_trainer_keys}

        if kwargs:
            for k, v in kwargs.items():
                if k in training_args_keys and k not in training_args_kwargs:
                    training_args_kwargs.update({k:v})
                elif k in sft_trainer_keys and k not in sft_trainer_kwargs:
                    sft_trainer_kwargs.update({k:v})
                else:
                    raise KeyError(f"Config '{k}' is invalid param.")

        # custom dataset 정의
        train_dataset = KoWav2Vec2Dataset(self.audio_dataset["train"], self.label_dataset["train"], processor=self.processor, training=True)
        if "test" in self.audio_dataset:
            eval_dataset = KoWav2Vec2Dataset(self.audio_dataset["test"], self.label_dataset["test"], processor=self.processor, training=False)
        else:
            eval_dataset = None

        data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding="longest")

        # TrainingArguments 및 SFTTrainer 생성
        training_args = self.__create_training_args(**training_args_kwargs)
        
        def wrapped_cm(pred):
            return compute_metrics(pred, self.processor)

        sft_trainer = self.__create_trainer(
            self.model,
            self.processor,
            train_dataset,
            eval_dataset,
            training_args=training_args,
            data_collator=data_collator,
            compute_metrics=wrapped_cm,
            **sft_trainer_kwargs
        )

        sft_trainer.train()

