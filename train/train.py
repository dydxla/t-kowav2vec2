# transformers import
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from transformers import TrainingArguments, Trainer

# Pytorch import
import torch

# Standard library import
import os

# source import
import data
import utils
import configs

def Train(data_type=None, tokenizer_dir=None, pre_model_dir=None, checkpoint_model_dir=None, output_dir=None, config_dir=None):
    if pre_model_dir:
        if tokenizer_dir:
            tokenizer_folder = os.path.join(os.path.dirname(__file__), '..', 'tokenizers')
            tokenizer_full_dir = os.path.join(tokenizer_folder, tokenizer_dir)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tokenizer_full_dir)
            processor = Wav2Vec2Processor.from_pretrained(pre_model_dir, tokenizer=tokenizer)
        else:
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pre_model_dir)
            processor = Wav2Vec2Processor.from_pretrained(pre_model_dir)
        model = Wav2Vec2ForCTC.from_pretrained(
            pre_model_dir,
            ignore_mismatched_sizes=True
        )
        model.lm_head = torch.nn.Linear(model.lm_head.in_features, len(tokenizer), bias=False)
        model.config.vocab_size = len(tokenizer)
        model.config.pad_token_id = tokenizer.pad_token_id
    else:    
        checkpoint_model_folder = os.path.join(os.path.dirname(__file__), '..', 'models', 'pre')
        checkpoint_model_full_dir = os.path.join(checkpoint_model_folder, checkpoint_model_dir)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(checkpoint_model_full_dir)
        processor = Wav2Vec2Processor.from_pretrained(checkpoint_model_full_dir)
        model = Wav2Vec2ForCTC.from_pretrained(
            checkpoint_model_full_dir,
            ignore_mismatched_sizes=True
        )
    model = model.half()
    
    data_dict = data.load_data(data_type)
    train_path_list = data_dict['train_path']
    train_label_list = data_dict['train_label']
    valid_path_list = data_dict['valid_path']
    valid_label_list = data_dict['valid_label']

    train_path_ktcloud_list = []
    valid_path_ktcloud_list = []
    for path in train_path_list:
        real_path = path.replace('/mnt/hda', '/home/work/iseollem')
        train_path_ktcloud_list.append(real_path)
        
    for path in valid_path_list:
        real_path = path.replace('/mnt/hda', '/home/work/iseollem')
        valid_path_ktcloud_list.append(real_path)

    train_dataset = utils.CustomDataset(train_path_ktcloud_list, train_label_list, processor)
    valid_dataset = utils.CustomDataset(valid_path_ktcloud_list, valid_label_list, processor, training=False)

    data_collator = utils.DataCollatorCTCWithPadding(processor=processor, padding="longest")
    
    training_args = configs.get_training_args(output_dir, config_dir)

    def wrapped_cm(pred):
        return utils.compute_metrics(pred, processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=wrapped_cm,
    )
    
    torch.cuda.empty_cache()
    trainer.train()