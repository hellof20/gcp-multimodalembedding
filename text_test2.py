from os import listdir
from os.path import isfile, join
from helper import get_text_embeddings
import chromadb

client = chromadb.PersistentClient(path="mydb")
collection = client.get_or_create_collection(name='text-test-004')
collection_num = collection.count()
content_cn = [
    "前几天，我看到有人介绍'饮水鸟'玩具，心痒痒，就从网上买了一个。价格很便宜，十几元人民币。到手以后，我玩了一下，觉得很有意思，分享给大家。它是一个物理学玩具，有点像永动机，只要把鸟头按到水杯里，让鸟嘴碰到冷水，它就会一直弹起、倒下、弹起、倒下......",
    "我买的那个，这样运动了一天一夜，都没有停。更有意思的是它的原理，你想不到可以这样用。首先，去掉那些故意迷惑人的外部装饰，它就是一个密封的异型玻璃容器。",
    "两头是圆球，中间由一根玻璃管相连，下部装了一些易挥发的液体（比如二氯甲烷、乙醚、酒精）。鸟嘴被毛毡包裹，保证冷水会充分附着在上面。",
    "遇到冷水以后，由于水温低，以及水份蒸发带走热量，使得容器上部的气压下降，下部的气压就会大于上部。下部的气压就会压着液体，让它顺着玻璃管上升，被压入上部。随着液体流入，鸟头越来越重，最终倒入水中，再次变成饮水的姿势。鸟身倾斜以后，玻璃管在鸟尾的一端就会露出水面，从而玻璃管两端的气压就会变得相等。由于重力的作用，上部的液体重新流回下部，从而鸟头再次弹起。这个过程会一直重复下去，只要冷水使得上部与下部之间存在温度差。本质上，饮水鸟是一个热机，不需要其他动力，靠温度差驱动。",
    "我觉得，这真是热量转换为能量的一个绝佳演示。但是，除了这个玩具，没听说过有其他的实际应用。我倒是看到过一篇报道 ，有人提出00108-X)，将饮水鸟做成发电机，利用温度差发电。虽然我猜测，发电效率一定很差，但如果实现了，一定很有趣。"
    ]

content_en = [
    "The length of the name must be between 3 and 63 characters.",
    "The name must start and end with a lowercase letter or a digit, and it can contain dots, dashes, and underscores in between.",
    "The name must not contain two consecutive dots.",
    "The name must not be a valid IP address.",
    "Most importantly, there is no default embedding function. If you add() documents without embeddings, you must have manually specified an embedding function and installed the dependencies for it."
]
content = content_cn
ids = ["1","2","3","4","5"]

if collection_num == 0:
    embeddings = get_text_embeddings(content=content)
    collection.add(
        embeddings = embeddings,
        documents = content,
        ids = ids
    )
print("embeddings num:",collection.count())

query_string = "将饮水鸟做成发电机，利用温度差发电"
embeddings = get_text_embeddings(content=[query_string])
result = collection.query(query_embeddings = embeddings, n_results = 2)
print(result)