import mtcnn
import numpy as np
from keras import backend as K
from keras.utils.data_utils import get_file
from scipy import spatial
import cv2
import numpy as np
import PIL
from google.cloud import storage
import functions_framework
import tensorflow as tf


model = None


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {"channels_last", "channels_first"}

    if version == 1:
        if data_format == "channels_first":
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == "channels_first":
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp


def generate_embeddings(face):

    face = preprocess_input(face, version=2)
    embeddings_known = model.predict(face)
    return embeddings_known


def compare(embedding_known, embedding_unknown, limit=0.45):
    dist = spatial.distance.cosine(
        embedding_known.flatten(), embedding_unknown.flatten()
    )
    if dist > limit:
        return 0
    else:
        return 1


@functions_framework.http
def handler(request):
    global model
    if request.method == "OPTIONS":
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }

        return ("", 204, headers)
    if model is None:
        download_blob("thecrapbucket", "tfmodels/faceai.h5", "/tmp/faceai.h5")
        model = tf.keras.models.load_model("/tmp/faceai.h5")

    request_json = request.get_json()
    true_face = request_json["truth"]
    check_face = request_json["check"]
    match = compare(generate_embeddings(true_face), generate_embeddings(check_face))
    if match == 0:
        match = "No"
    else:
        match = "Yes"

    result = {"Match": match}
    # Set CORS headers for the main request
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST",
    }

    return (result, 200, headers)
