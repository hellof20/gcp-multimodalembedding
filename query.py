from os import listdir
from os.path import isfile, join
from helper import get_multi_text_embedding, get_multi_image_embedding
import chromadb

client = chromadb.PersistentClient(path="mydb")
collection = client.get_or_create_collection(name='video-test')

## 通过文本查询文本
image='images-2/sjbb.jpeg'
embedding = get_multi_image_embedding(image_path = image)
result = collection.query(
    query_embeddings = [embedding],
    n_results=2
    )
print(result)

## 通过图片查询图片
# image='images-2/sjbb.jpeg'
# embedding = get_multi_image_embedding(image_path = image)
# result = collection.query(
#     query_embeddings = [embedding],
#     n_results=2
#     )
# print(result)

## 通过文本查询图片
# query_string = "手表"
# embedding = get_multi_text_embedding(text=query_string)
# result = collection.query(
#     query_embeddings = embedding,
#     n_results=1
#     )
# print(result)

## 通过文本查询视频中内容
query_string = "士兵冲锋对战"
embedding = get_multi_text_embedding(text=query_string )
result = collection.query(
    query_embeddings = embedding,
    n_results=2
    )
print(result['ids'])