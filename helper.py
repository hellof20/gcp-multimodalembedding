import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.vision_models import Image, Video, MultiModalEmbeddingModel, VideoSegmentConfig
import os

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


def get_multi_text_image_embedding(text, image_path):
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    image = Image.load_from_file(image_path)
    embeddings = model.get_embeddings(
        image = image,
        contextual_text=text
    )
    return embeddings.image_embedding, embeddings.text_embedding


def get_multi_video_embedding(video_path, interval_sec):
    video_embeddings = []
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    response = video_duration_size(video_path)
    video_size = response['format_size']
    if video_size > 25:
        print('视频压缩中...')
        video_path = video_pre_process(video_path)
    video = Video.load_from_file(video_path)
    video_segment_config = VideoSegmentConfig()
    video_segment_config.interval_sec = interval_sec
    # print(video_segment_config.interval_sec)
    embeddings = model.get_embeddings(
        video = video,
        video_segment_config = video_segment_config
    )
    i = 1
    for video_embedding in embeddings.video_embeddings:
        print(f"Video Segment {i}: {video_embedding.start_offset_sec} - {video_embedding.end_offset_sec}")
        video_embeddings.append(video_embedding.embedding)
        i = i + 1
    return video_embeddings


def get_text_embeddings(content):
    # "textembedding-gecko-multilingual@001" "text-embedding-preview-0409"
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual@001")
    task = "RETRIEVAL_DOCUMENT"
    inputs = [TextEmbeddingInput(text, task) for text in content]
    embeddings = model.get_embeddings(inputs)
    result = [embedding.values for embedding in embeddings]
    dimensions = len(result[0])
    # print("dimensions is %s" % str(dimensions))
    return result


def video_duration_size(path):
    try:
        command1 = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {}".format(path)
        duration = float(os.popen(command1).read().strip())
    except:
        command1 = "ffmpeg.ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {}".format(path)
        duration = float(os.popen(command1).read().strip())
    size = os.stat(path)
    format_size = round(size.st_size/(1024*1024),2)
    print("- video size: {} MB".format(format_size))
    print("- video duration: {} Seconds".format( round(duration,2) ))
    respone = {"duration": duration, "format_size": format_size}
    return respone


def video_pre_process(video_path):
    dir = 'data/'
    video_name = video_path.split('/')[1].split('.')[0]
    output_file_path = dir + '_' + video_name + '.mp4'
    if not os.path.exists(output_file_path):
        os.system('ffmpeg -i {} -an -crf 32 -r 24 -y -loglevel quiet {}'.format(video_path, output_file_path))
        video_duration_size(output_file_path)
    return output_file_path


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