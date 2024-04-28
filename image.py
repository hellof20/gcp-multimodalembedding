from os import listdir
from os.path import isfile, join
from helper import get_multi_image_embedding, get_multi_text_embedding
import chromadb

client = chromadb.PersistentClient(path="mydb")
collection = client.get_or_create_collection(name='collection1')
collection_num = collection.count()

if collection_num == 0:
    path='data'
    image_embeddings,text_embeddings = [],[]
    image_ids, text_ids = [],[]
    files = listdir(path)
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = path + '/' + file
            image_embedding = get_multi_image_embedding(image_path = image_path)
            image_embeddings.append(image_embedding)
            id = file.split('.')[0]
            image_ids.append(id)
        # if file.endswith(('.txt')):
        #     with open(path + '/' + file, 'r') as f:
        #         text = f.read()
        #     text_embedding = get_multi_text_embedding(text)
        #     text_embeddings.append(text_embedding)
        #     id = file.split('.')[0]
        #     text_ids.append(id)

    # write embeddings to vector store
    collection.add(embeddings = image_embeddings,ids = image_ids)
    # collection.add(embeddings = text_embeddings,ids = text_ids)

print("Total embeddings num:",collection.count())


def text_query(text, num):
    embedding = get_multi_text_embedding(text = text)
    result = collection.query(
        query_embeddings = embedding,
        n_results=num)
    return result


def image_query(image_path, num):
    embedding = get_multi_image_embedding(image_path = image_path)
    result = collection.query(
        query_embeddings = [embedding],
        n_results = num)
    return result


print(text_query('bag',6))
# print(image_query('data/p001.jpeg',6))