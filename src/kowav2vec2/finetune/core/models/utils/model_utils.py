import torch


def model_preprocess(model, tokenizer):
    model.lm_head = torch.nn.Linear(model.lm_head.in_features, len(tokenizer), bias=False)
    model.config.vocab_size = len(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model