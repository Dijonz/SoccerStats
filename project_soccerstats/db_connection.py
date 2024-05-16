import pymongo
from pymongo import MongoClient

client = MongoClient("mongodb://hcortezi10:9voCKDy8spievy8R@ac-cdnusv3-shard-00-00.savault.mongodb.net:27017,ac-cdnusv3-shard-00-01.savault.mongodb.net:27017,ac-cdnusv3-shard-00-02.savault.mongodb.net:27017/?replicaSet=atlas-nx5y6z-shard-0&ssl=true&authSource=admin") #connection string
db = client['soccerstats']