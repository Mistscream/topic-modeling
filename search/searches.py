from spacy_preprocessing.preprocess import Preprocess

from .data import texts, titles, ids, urls, mongo_collection
from .models import id2word, lda_model, lsi_model, index
from .preprocessing import preprocess_after, get_name_entities

import pymongo


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
            'named_entities': get_named_entities(result[2]),
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
            'named_entities': get_named_entities(result[2]),
            'url': result[5]
        }
        for result in similarity_list[:max_results]
    ]


def fulltext_search(word, max_results=10):
    results = mongo_collection.find( 
        { '$text': { '$search': word, '$language': 'de' } },
        { 'score': { '$meta': 'textScore' } }
    ).sort([
        ('score', { '$meta': 'textScore' })
    ]).limit(max_results)

    return [
        {
            'id': str(result['_id']),
            'score': float(result['score']),
            'title': result['title'],
            'text': result['text'],
            'named_entities': get_named_entities(result['text']),
            'url': result['url']
        }
        for result in results
    ]
