from os import listdir
from os.path import isfile, join
from helper import get_multi_video_embedding, get_multi_text_embedding, get_multi_image_embedding
import chromadb

client = chromadb.PersistentClient(path="mydb")
collection = client.get_or_create_collection(name='video')
collection_num = collection.count()
data_dir = 'data/'


# single video embedding and write to vectordb 
def single_video_embedding(video_name):
    ids = []
    video_embeddings = []
    video_path = data_dir + video_name
    embeddings = get_multi_video_embedding(video_path, 5)
    for i in range(1, len(embeddings)+1):
        ids.append(video_name.split('.')[0] + '_' + str(i))
    video_embeddings.extend(embeddings)
    # write video embeddings to vector store
    collection.add(embeddings = video_embeddings, ids = ids)
    print("Totle Video embeddings num:", collection.count())


def text_query_video(text, num):
    embedding = get_multi_text_embedding(text=text)
    result = collection.query(
        query_embeddings = embedding,
        n_results = num
        )
    return result['ids'][0]


def image_query_video(image_name, num):
    image_path = data_dir + image_name
    embedding = get_multi_image_embedding(image_path = image_path)
    result = collection.query(
        query_embeddings = embedding,
        n_results = num
        )
    return result['ids'][0]


# batch videos embedding and write to vectordb 
def batch_videos_embedding():
    files = listdir(data_dir)
    for file in files:
        if file.endswith(('.mp4')):
            print(file)
            single_video_embedding(file)

# batch_videos_embedding()
# single_video_embedding('f00b3544-4bcc-11ee-af07-26cd4c413080.mp4')

# print(text_query_video("A Woman and Three Infantrymen", 4))
# print(text_query_video("狙击枪", 4))
# print(text_query_video("万箭齐发", 4))
# print(text_query_video("A man walks on the sofa with a fan in his hand", 6))
# print(image_query_video("贝琳达.jpg", 4))
# print(image_query_video("傲血雄狮.png", 4))
# print(image_query_video("士兵.png", 4))