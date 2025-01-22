import evaluate
import numpy as np

cer_metric = evaluate.load("cer")

def compute_metrics(pred, processor):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    for i in range(pred_ids.shape[0]):
        for j in range(pred_ids.shape[1]):
            if pred_logits[i, j, 0] == -100.0:
                pred_ids[i, j] = processor.tokenizer.pad_token_id

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}