import pymongo
from pymongo import MongoClient
import pprint

clint = MongoClient()
db = clint.Users  #DataBase
collection = db.user
print('Database Names:', clint.list_database_names())
print('Collection(s) Name: ', db.list_collection_names())
for i in collection.find_one({}):
    print(i)
print(collection.find_one({}))
# insert

data = {"name": "Python", "to": "Mongo DB"}

collection.insert_one(data)

print(collection.find_one({"name": "Python"}))