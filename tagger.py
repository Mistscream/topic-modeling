from tqdm import tqdm
from search.data import mongo_collection
from search.preprocessing import get_named_entities, get_lstm_embedding, get_pooling_embedding


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