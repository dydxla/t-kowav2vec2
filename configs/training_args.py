from transformers import TrainingArguments
import os

def get_training_args(output_dirname, config_dirname):
    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__), '..', 'models', f'{output_dirname}'),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        # eval_accumulation_steps=8,
        learning_rate=0.0001,
        weight_decay=3e-7,
        warmup_steps=10000,
        max_steps=60000,
        fp16=True,
        deepspeed=os.path.join(os.path.dirname(__file__), '..', 'configs', f'{config_dirname}'),
        evaluation_strategy="steps",
        save_steps=2000,
        eval_steps=1000,
        logging_steps=1000,
        metric_for_best_model="cer",
        logging_dir=os.path.join(os.path.dirname(__file__), '..', 'models', f'{output_dirname}', 'logs'),
        greater_is_better=False,
        push_to_hub=False,
    )
    
    return training_args