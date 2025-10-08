from grapheme import graphemes

def get_char_ngrams(text, n):
    chars = list(graphemes(text))
    return ["".join(chars[i:i+n]) for i in range(len(chars)-n+1)]

def get_word_ngrams(text, n):
    words = text.split()
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

def ngram_overlap(candidate_ngrams, reference_ngrams):
    overlap = 0
    ref_tmp = reference_ngrams.copy()
    for ng in candidate_ngrams:
        if ng in ref_tmp:
            overlap += 1
            ref_tmp.remove(ng)
    return overlap

def chrf_score_sentence(candidate, reference, max_char_n=6, max_word_n=2):
    """
    문장 단위 ChrF++ Precision, Recall 합산 결과 반환
    """
    total_prec, total_rec, count = 0.0, 0.0, 0
    eps = 1e-12

    # 문자 n-gram
    for n in range(1, max_char_n+1):
        cand_ngrams = get_char_ngrams(candidate, n)
        ref_ngrams = get_char_ngrams(reference, n)

        if not cand_ngrams or not ref_ngrams:
            continue

        overlap = ngram_overlap(cand_ngrams, ref_ngrams)
        prec = overlap / (len(cand_ngrams) + eps)
        rec = overlap / (len(ref_ngrams) + eps)

        total_prec += prec
        total_rec += rec
        count += 1

    # 단어 n-gram
    for n in range(1, max_word_n+1):
        cand_ngrams = get_word_ngrams(candidate, n)
        ref_ngrams = get_word_ngrams(reference, n)

        if not cand_ngrams or not ref_ngrams:
            continue

        overlap = ngram_overlap(cand_ngrams, ref_ngrams)
        prec = overlap / (len(cand_ngrams) + eps)
        rec = overlap / (len(ref_ngrams) + eps)

        total_prec += prec
        total_rec += rec
        count += 1

    return total_prec, total_rec, count

def chrf_corpus(candidates, references, max_char_n=6, max_word_n=2, beta=2):
    """
    Corpus-level ChrF++ 점수 계산 (Precision, Recall, F1 반환)
    """
    total_prec, total_rec, total_count = 0.0, 0.0, 0

    for cand, ref in zip(candidates, references):
        p, r, c = chrf_score_sentence(cand, ref, max_char_n, max_word_n)
        total_prec += p
        total_rec += r
        total_count += c

    avg_prec = total_prec / (total_count + 1e-12)
    avg_rec = total_rec / (total_count + 1e-12)

    beta2 = beta**2
    f1 = (1 + beta2) * (avg_prec * avg_rec) / (beta2 * avg_prec + avg_rec + 1e-12)

    return {
        "precision": avg_prec * 100,
        "recall": avg_rec * 100,
        "f1": f1 * 100
    }

import evaluate

def chrf_corpus_evaluate(predictions, references):
    chrf_metric = evaluate.load("chrf")
    chrf = chrf_metric.compute(predictions=predictions, references=references)['score']
    return chrf

def chrf_plus_evaluate(predictions, references):
    chrf_plus_metric = evaluate.load("chrf")
    chrf_plus = chrf_plus_metric.compute(predictions=predictions, references=references, word_order=2)['score']
    return chrf_plus