from gensim import corpora,similarities
from gensim.models import LdaModel, LsiModel

lda_model = LdaModel.load('./data/lda_model')
lsi_model = LsiModel.load('./data/lsi_model')
id2word = corpora.Dictionary.load('./data/id2word')
index = similarities.MatrixSimilarity.load('./data/index')