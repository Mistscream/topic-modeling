import os
import pymongo
from pprint import pprint
import gensim
from gensim import corpora, models, similarities
from spacy_preprocessing.preprocess import Preprocess

import numpy as np

print(os.environ['POLICE_REPORTS_MONGO_URI'])

client = pymongo.MongoClient(os.environ['POLICE_REPORTS_MONGO_URI'])

db = client[os.environ['POLICE_REPORTS_MONGO_DATABASE']]
# print(db)

collection = db[os.environ['POLICE_REPORTS_MONGO_COLLECTION']]
# print(collection)
# pprint.pprint(collection.find_one())

items = [item for item in collection.find().limit(10000)]
texts = [report['text'] for report in items]
reports = [report['text_pre_processed_v1'] for report in items]
# reports = [report['text_pre_processed_v1'] for report in collection.find()]
# pprint.pprint(reports)

dictionary = corpora.Dictionary(reports)

# print(dict.token2id)

corpus = [dictionary.doc2bow(report) for report in reports]

# corpus_report = [corpus[7]]
# pprint.pprint(corpus)

lsi = models.LsiModel(corpus, num_topics=10, id2word=dictionary)
# print(lsi)

index = similarities.MatrixSimilarity(lsi[corpus])
# print(index.get_similarities())

# pprint(len(lsi.get_topics()[0]))
# pprint(lsi.print_topics())

search_term = "auto unfall frau"
preprocess = Preprocess(search_term)
search_term_preprocessed = preprocess.preprocess(sentence_split=False, with_pos=False)
# print(search_term_preprocessed)
search_term_bow = dictionary.doc2bow(search_term_preprocessed)
search_term_lsi = lsi[search_term_bow]
# print(search_term_lsi)

sims = index[search_term_lsi]
# max = np.argmax(sims)
# print(max)
# print(texts[max])

similarity_list = list(zip(range(len(sims)), sims, texts))
similarity_list.sort(key=lambda x: x[1], reverse=True)
pprint(similarity_list[:10])
