import os
import pickle
import pymongo

client = pymongo.MongoClient(os.environ['POLICE_REPORTS_MONGO_URI'])
db = client[os.environ['POLICE_REPORTS_MONGO_DATABASE']]
collection = db[os.environ['POLICE_REPORTS_MONGO_COLLECTION']]
items = [item for item in collection.find()]
pickle.dump(items, open('./data/items.pkl', 'wb'))