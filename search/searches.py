from spacy_preprocessing.preprocess import Preprocess

from .data import texts, titles, ids, urls
from .models import id2word, lda_model, lsi_model, index
from .preprocessing import preprocess_after


def lda_search(word, max_results=10):
    preprocess = Preprocess(word)
    query_preprocessed = preprocess_after(preprocess.preprocess(sentence_split=False, with_pos=False))
    query_bow = id2word.doc2bow(query_preprocessed)
    query_vec = lda_model[query_bow]

    sims = index[query_vec]
    similarity_list = list(zip(range(len(sims)), sims, texts, titles, ids, urls))
    similarity_list.sort(key=lambda x: x[1], reverse=True)

    return [
        {
            'id': result[0],
            'score': float(result[1]),
            'title': result[3],
            'text': result[2],
            'url': result[5]
        }
        for result in similarity_list[:max_results]
    ]


def lsi_search(word, max_results=10):
    preprocess = Preprocess(word)
    query_preprocessed = preprocess_after(preprocess.preprocess(sentence_split=False, with_pos=False))
    query_bow = id2word.doc2bow(query_preprocessed)

    query_vec = lsi_model[query_bow]

    sims = index[query_vec]
    similarity_list = list(zip(range(len(sims)), sims, texts, titles, ids, urls))
    similarity_list.sort(key=lambda x: x[1], reverse=True)

    return [
        {
            'id': result[0],
            'score': float(result[1]),
            'title': result[3],
            'text': result[2],
            'url': result[5]
        }
        for result in similarity_list[:max_results]
    ]


def fulltext_search(word, max_results=10):
    return word
