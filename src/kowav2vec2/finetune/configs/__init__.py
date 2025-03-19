import os

_conf = dict(
    mix_noise_ratio=0.5,
    sampling_rate=16000,
    audio_format="wav",
    label_format="txt",
    half_model=True
)

_train_args = dict(
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
push_to_hub=False
)

_train_conf = {

}
