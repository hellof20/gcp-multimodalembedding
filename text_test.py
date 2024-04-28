from os import listdir
from os.path import isfile, join
from helper import get_multi_text_embedding
import chromadb

client = chromadb.PersistentClient(path="mydb")
collection = client.get_or_create_collection(name='text-test-001')
collection_num = collection.count()

content = [
    "The length of the name must be between 3 and 63 characters.",
    "The name must start and end with a lowercase letter or a digit, and it can contain dots, dashes, and underscores in between.",
    "The name must not contain two consecutive dots.",
    "The name must not be a valid IP address.",
    "Most importantly, there is no default embedding function. If you add() documents without embeddings, you must have manually specified an embedding function and installed the dependencies for it."
]
ids = ["1","2","3","4","5"]
embeddings = []
if collection_num == 0:
    for text in content:
        embedding = get_multi_text_embedding(text=text)
        embeddings.append(embedding)
    collection.add(
        embeddings = embeddings,
        documents = content,
        ids = ids
    )
    print("embeddings num:",collection.count())

query_string = "将饮水鸟做成发电机，利用温度差发电"
embedding = get_multi_text_embedding(text=query_string)
result = collection.query(query_embeddings = [embedding],n_results=2)
print(result)