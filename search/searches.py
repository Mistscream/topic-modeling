from spacy_preprocessing.preprocess import Preprocess

from .data import texts, titles, ids, urls, mongo_collection
from .models import id2word, lda_model, lsi_model, index
from .preprocessing import preprocess_after, get_named_entities, get_lstm_embedding, get_pooling_embedding
from scipy.spatial.distance import cosine

import pymongo
import numpy as np


def lda_search(query, max_results=10):
    preprocess = Preprocess(query)
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


def lsi_search(query, max_results=10):
    preprocess = Preprocess(query)
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


def fulltext_search(query, max_results=10):
    results = mongo_collection.find( 
        { '$text': { '$search': query, '$language': 'de' } },
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

def flair_pooling_search(query, max_results=10):
    cursor = mongo_collection.find({ 'text_pooling_embedding': { '$exists': True } })
    query_embedding = get_pooling_embedding(query)

    results = [
        {
            'id': str(record['_id']),
            'score': cosine(query_embedding, record['text_pooling_embedding']) / 2.0,
            'title': record['title'],
            'text': record['text'],
            'url': record['url']
        }
        for record in cursor
    ]

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

def flair_lstm_search(query, max_results=10):
    cursor = mongo_collection.find({ 'text_lstm_embedding': { '$exists': True } })
    query_embedding = get_lstm_embedding(query)

    results = [
        {
            'id': str(record['_id']),
            'score': cosine(query_embedding, record['text_lstm_embedding']) / 2.0,
            'title': record['title'],
            'text': record['text'],
            'url': record['url']
        }
        for record in cursor
    ]

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]