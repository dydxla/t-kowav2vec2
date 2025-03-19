from kowav2vec2.finetune.configs import _train_args, _train_conf

class BaseTrainer:
    
    TRAIN_ARGS = _train_args
    TRAIN_PARAMS = _train_conf

    def __init__(
            self, 
            initial_configs: dict = None,
    ):
        self.configs = {
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 8,
            "learning_rate": 3e-7,
            "evaluation_strategy": "steps",
            "max_steps": 6000,
            "metric_for_best_model": "cer"
        }

        if not initial_configs:
            self.configs = {**self.TRAIN_ARGS, **self.TRAIN_PARAMS}
        else:
            self.configs.update(initial_configs)

    def __repr__(self,):
        return f"BaseTrainer(configs={self.configs})"
    
    def update_config(self, key, value):
        if key in self.configs:
            self.configs[key] = value
        else:
            raise KeyError(f"Config '{key}' does not exist in configs.")
        
    def get_config(self, key):
        return self.configs.get(key, None)

    def show_configs(self):
        return self.configs