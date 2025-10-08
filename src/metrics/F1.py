#sentence f1
def f1_score_sentence(sentences_pred, sentences_true):
    """
    문장 단위 F1 점수 계산
    :param sentences_pred: 예측된 문장 리스트
    :param sentences_true: 실제 문장 리스트
    :return: F1 점수
    """
    if len(sentences_pred) != len(sentences_true):
        raise ValueError("Predicted and true sentences must have the same length.")

    total_f1 = 0.0
    for pred, true in zip(sentences_pred, sentences_true):
        pred_set = set(pred.split())
        true_set = set(true.split())

        tp = len(pred_set & true_set)  # True Positives
        fp = len(pred_set - true_set)  # False Positives
        fn = len(true_set - pred_set)  # False Negatives

        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

        total_f1 += f1

    average_f1 = total_f1 / len(sentences_pred) if sentences_pred else 0.0
    
    return average_f1 * 100