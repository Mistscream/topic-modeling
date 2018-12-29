import os
import pymongo
from pprint import pprint
import gensim
from gensim import corpora, models, similarities
from spacy_preprocessing.preprocess import Preprocess

import pickle
import numpy as np

items = pickle.load(open('./data/items.pkl', 'rb'))

texts = [report['text'] for report in items]
reports = [report['text_pre_processed_v1'] for report in items]
# reports = [report['text_pre_processed_v1'] for report in collection.find()]
# pprint.pprint(reports)

reports_dict = corpora.Dictionary(reports)

# print(dict.token2id)

corpus = [reports_dict.doc2bow(report) for report in reports]

# corpus_report = [corpus[7]]
# pprint.pprint(corpus)

lsi = models.LsiModel(corpus, num_topics=10, id2word=reports_dict)
# print(lsi)

index = similarities.MatrixSimilarity(lsi[corpus])
# print(index.get_similarities())

# pprint(len(lsi.get_topics()[0]))
# pprint(lsi.print_topics())

search_term = "auto unfall frau"
preprocess = Preprocess(search_term)
search_term_preprocessed = preprocess.preprocess(sentence_split=False, with_pos=False)
# print(search_term_preprocessed)
search_term_bow = reports_dict.doc2bow(search_term_preprocessed)
search_term_lsi = lsi[search_term_bow]
# print(search_term_lsi)

sims = index[search_term_lsi]
# max = np.argmax(sims)
# print(max)
# print(texts[max])

similarity_list = list(zip(range(len(sims)), sims, texts))
similarity_list.sort(key=lambda x: x[1], reverse=True)
pprint(similarity_list[:10])
