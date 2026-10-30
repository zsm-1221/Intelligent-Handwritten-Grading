import re
import unicodedata

def edit_distance(a,b):
    row = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt = [i]
        for j,y in enumerate(b,1):
            nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row = nxt
    return row[-1]

def normalize_ocr(text):
    return " ".join(unicodedata.normalize("NFKC",text).lower().split())

def error_counts(reference, prediction):
    ref, hyp = normalize_ocr(reference), normalize_ocr(prediction)
    return {"char_errors":edit_distance(ref,hyp), "reference_chars":len(ref),
            "word_errors":edit_distance(ref.split(),hyp.split()), "reference_words":len(ref.split())}

def aggregate_errors(rows):
    total = {k:sum(r[k] for r in rows) for k in ("char_errors","reference_chars","word_errors","reference_words")}
    return {"n":len(rows), **total, "cer":total["char_errors"]/total["reference_chars"] if total["reference_chars"] else None,
            "wer":total["word_errors"]/total["reference_words"] if total["reference_words"] else None,
            "normalization":"Unicode NFKC, lowercase, whitespace collapsed; punctuation retained; corpus micro average."}
