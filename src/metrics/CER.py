import evaluate

def calculate_cer_sentences(predictions, references):
    cer_metric = evaluate.load("cer")
    cer = cer_metric.compute(predictions=predictions, references=references)
    return cer*100