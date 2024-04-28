from os import listdir
from os.path import isfile, join
from helper import get_multi_text_image_embedding, get_multi_text_embedding
import chromadb

client = chromadb.PersistentClient(path="mydb")
collection = client.get_or_create_collection(name='collection1')
collection_num = collection.count()
image_embeddings,text_enbeddings = [],[]
images_path = ['data/p001.jpeg','data/p003.jpg']
content = ['a picture of huge', 'a picture of liudehua']
image_ids = ['1','2']
text_ids = ['3','4']

if collection_num == 0:
    for i in range(0,2):
        image_embedding, text_embedding = get_multi_text_image_embedding(content[i], images_path[i])
        image_embeddings.append(image_embedding)
        text_enbeddings.append(text_embedding)

    # # write embeddings to vector store
    collection.add(embeddings = image_embeddings,ids = image_ids)
    collection.add(embeddings = text_enbeddings,ids = text_ids)
print("Total embeddings num:",collection.count())


def text_query(text, num):
    embedding = get_multi_text_embedding(text = text)
    result = collection.query(
        query_embeddings = embedding,
        n_results=num)
    return result

print(text_query('穿西服的男人',4))