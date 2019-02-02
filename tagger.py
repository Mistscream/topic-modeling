import spacy

from tqdm import tqdm
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, DocumentLSTMEmbeddings, Sentence
from search.data import mongo_collection
from search.preprocessing import get_named_entities

nlp = spacy.load('de')
glove_embedding = WordEmbeddings('de')
flair_embedding_forward = FlairEmbeddings('german-forward')
flair_embedding_backward = FlairEmbeddings('german-backward')

document_pooling_embeddings = DocumentPoolEmbeddings([
    glove_embedding,
    flair_embedding_backward,
    flair_embedding_forward
])

document_lstm_embeddings = DocumentLSTMEmbeddings([
    glove_embedding,
    flair_embedding_backward,
    flair_embedding_forward
])

def get_lstm_embedding(document):
    tokens = [token.text for token in nlp(document)]
    text = ' '.join(tokens)
    sentence = Sentence(text)
    document_lstm_embeddings.embed(sentence)

    return sentence.get_embedding().squeeze().tolist()    

def get_pooling_embedding(document):
    tokens = [token.text for token in nlp(document)]
    text = ' '.join(tokens)
    sentence = Sentence(text)
    document_pooling_embeddings.embed(sentence)

    return sentence.get_embedding().squeeze().tolist()


cursor = mongo_collection.find({ 'named_entities': { '$exists': False }})
tqdm_cursor = tqdm(cursor, total=cursor.count())

# records_with_embeddings = [ 
#    {
#            'id': str(record['_id']),
#            'title': record['title'],
#            'text': record['text'],
#            'category': record['category'],
#            'named_entities': get_name_entities(record['text']),
#            'text_pooling_embedding': get_pooling_embedding(record['text']),
#            'text_lstm_embedding': get_lstm_embedding(record['text']),
#            'url': record['url']
#    }
#    for record in tqdm_cursor
# ]

for record in tqdm_cursor:
    mongo_collection.update_one(
        { '_id': record['_id'] }, 
        { '$set': {
             'named_entities': get_named_entities(record['text']),
             'text_pooling_embedding': get_pooling_embedding(record['text']),
             'text_lstm_embedding': get_lstm_embedding(record['text'])
        } }
    )