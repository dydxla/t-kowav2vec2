from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer

def load_processor(
        model_dir: str,
        tokenizer = None
):
    processor = Wav2Vec2Processor.from_pretrained(model_dir, tokenizer=tokenizer)
    return processor


def load_tokenizer_and_processor(
        model_dir: str = None, 
        tokenizer_dir: str = None
    ):
    if tokenizer_dir:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tokenizer_dir)
        processor = load_processor(model_dir, tokenizer)
    else:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_dir)
        processor = load_processor(model_dir)
    
    return tokenizer, processor

