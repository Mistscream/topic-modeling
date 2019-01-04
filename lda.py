from pprint import pprint
import gensim
from gensim import corpora, models, similarities
from gensim.models import LdaModel
from spacy_preprocessing.preprocess import Preprocess

import numpy as np
import pickle

blacklist = ['polizei', 'polizist', 'beamter', 'nr.', 'berlin', 'uhr', 'polizeimeldung',
             'nicht', 'jahr', 'jährige', 'jährig', 'jähriger', 'polizeiliche', 'polizeilich', '2015', '2016',
             '2014', '2017', '2018', 'polizeibeamter', '-', '  ', ' ', '   ', '    ']

items = pickle.load(open('./data/items.pkl', 'rb'))
data = [report['text_pre_processed_v1'] for report in items]

clean_data = []
for doc in data:
    clean_data.append([word for word in doc if word not in blacklist])

id2word = corpora.Dictionary(clean_data)
# pprint(list(id2word.items())[0:20])
corpus = [id2word.doc2bow(doc) for doc in clean_data]
# pprint(corpus[0:10])

# Human readable format of corpus (term-frequency)
# pprint([[(id2word[id], freq) for id, freq in cp] for cp in corpus[1:2]])

# build a LDA model
lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=10, passes=15, alpha=1, eta=1)
pprint(lda_model.print_topics())