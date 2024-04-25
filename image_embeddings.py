from os import listdir
from os.path import isfile, join
from helper import get_multi_image_embedding
import chromadb

client = chromadb.PersistentClient(path="mydb")
collection = client.get_or_create_collection(name='image-test')

## 图片生成embedding写入ChromaDB
path='images-1'
image_embeddings = []
ids = []
images = [f for f in listdir(path) if isfile(join(path, f))]
for image in images:
    id = image.split('.')[0]
    ids.append(id)
    image_embeddings.append(get_multi_image_embedding(image_path = path + '/' + image))

collection.add(
    embeddings = image_embeddings,
    ids = ids
)
print("Images embeddings num:",collection.count())