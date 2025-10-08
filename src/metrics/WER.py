import evaluate

def calculate_wer_sentences(predictions, references):
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references)
    return wer*100