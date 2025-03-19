from transformers import Wav2Vec2ForCTC
from kowav2vec2.finetune.core.models.utils import model_preprocess

def load_wav2vec2(model_dir):
    model = Wav2Vec2ForCTC.from_pretrained(
        model_dir,
        ignore_mismatched_sizes=True
    )
    return model
    