from os import listdir
from os.path import isfile, join
from helper import get_multi_video_embedding
import chromadb

client = chromadb.PersistentClient(path="mydb")
collection = client.get_or_create_collection(name='video-test')

## 视频生成embedding写入ChromaDB
video_path = 'videos-1/rok_ad.mp4'
video_name = 'rok_ad'
embeddings = get_multi_video_embedding(video_path = video_path)
ids = []
for i in range(1, len(embeddings)+1):
    ids.append(video_name+'_'+str(i))
result = collection.add(
    embeddings = embeddings,
    ids = ids
    )
print("Video embeddings num:",collection.count())