import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.vision_models import Image, Video, MultiModalEmbeddingModel, VideoSegmentConfig

vertexai.init(project='speedy-victory-336109', location='us-central1')


def get_multi_image_embedding(image_path):
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    if image_path != None:
        image = Image.load_from_file(image_path)
        embeddings = model.get_embeddings(
            image=image
        )
    # print(f"Image Embedding: {embeddings.image_embedding}")
    # print(f"Text Embedding: {embeddings.text_embedding}")
    return embeddings.image_embedding

def get_multi_text_embedding(text):
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    embeddings = model.get_embeddings(
        contextual_text=text
    )
    return embeddings.text_embedding

def get_multi_video_embedding(video_path):
    video_embeddings = []
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    video = Video.load_from_file(video_path)
    video_segment_config = VideoSegmentConfig({"start_offset_sec":20,"interval_sec": 4})
    embeddings = model.get_embeddings(
        video=video,
        video_segment_config=video_segment_config
    )
    for video_embedding in embeddings.video_embeddings:
        print(f"Video Segment: {video_embedding.start_offset_sec} - {video_embedding.end_offset_sec}")
        video_embeddings.append(video_embedding.embedding)
    return video_embeddings


# def get_text_embeddings(text_model, content, task):
#     # task = "RETRIEVAL_DOCUMENT"
#     dimensionality = 512
#     # model = TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual@001")
#     # "text-embedding-preview-0409"
#     model = TextEmbeddingModel.from_pretrained(text_model)
#     inputs = [TextEmbeddingInput(text, task) for text in content]
#     kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
#     if text_model == "textembedding-gecko-multilingual@001":
#         embeddings = model.get_embeddings(inputs)
#     else:
#         embeddings = model.get_embeddings(inputs, **kwargs)
#     result = [embedding.values for embedding in embeddings]
#     dimensions = len(result[0])
#     print("dimensions is %s" % str(dimensions))
#     return result


# def query_embedding(collection, embedding):
#     if len(embedding) > 1:
#         query_embeddings = [embedding]
#     else:
#         query_embeddings = embedding
#     result = collection.query(
#         query_embeddings = query_embeddings,
#         n_results=1
#     )
#     return result
