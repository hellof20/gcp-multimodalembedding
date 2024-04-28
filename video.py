from os import listdir
from os.path import isfile, join
from helper import get_multi_video_embedding, get_multi_text_embedding
import chromadb

client = chromadb.PersistentClient(path="mydb")
collection = client.get_or_create_collection(name='video')
collection_num = collection.count()


if collection_num == 0:
    path='data'
    video_embeddings = []
    ids = []
    files = listdir(path)
    for file in files:
        if file.endswith(('.mp4')):
            print(file)
            video_path = path + '/' + file
            embeddings = get_multi_video_embedding(video_path)
            for i in range(1, len(embeddings)+1):
                ids.append(file.split('.')[0] + '_' + str(i))
            video_embeddings.extend(embeddings)
    # write video embeddings to vector store
    collection.add(embeddings = video_embeddings, ids = ids)

print("Video embeddings num:",collection.count())

def text_query_video(text, num):
    embedding = get_multi_text_embedding(text=text)
    result = collection.query(
        query_embeddings = embedding,
        n_results = num
        )
    return result['ids'][0]

print(text_query_video("有人坐在沙发上", 2))
# print(text_query_video("boat", 2))