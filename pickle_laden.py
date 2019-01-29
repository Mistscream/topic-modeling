import os
import pickle
import pymongo

# client = pymongo.MongoClient('mongodb://hoppe:abc123@ds113765.mlab.com:13765/hoppe')
# db = client['hoppe']
# collection = db['hoppe']
# items = [item for item in collection.find()]
# pickle.dump(items, open('./data/hoppe.pkl', 'wb'))


hoppe = pickle.load(open('./data/hoppe.pkl', 'rb'))
print(len(hoppe))