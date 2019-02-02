import pickle
from pymongo import MongoClient

items = pickle.load(open('./data/items.pkl', 'rb'))
texts = [report['text'] for report in items]
titles = [report['title'] for report in items]
ids = [report['_id'] for report in items]
urls = [report['url'] for report in items]
texts_pre_processed = [report['text_pre_processed_v1'] for report in items]


mongo_client = MongoClient('mongodb://hoppe:abc123@ds113765.mlab.com:13765/hoppe')
mongo_db = mongo_client['hoppe']
mongo_collection = mongo_db['hoppe']