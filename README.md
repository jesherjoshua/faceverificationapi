# FaceVerification API

This is a Python-based API for face verification using an embedding-based methodology and ResNet50 to create the embeddings. The API consists of two main files: `main.py` and `requirements.txt`.

## Installation

To install the required dependencies, run the following command:

<pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg" data-darkreader-inline-stroke=""><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs">pip install -r requirements.txt
</code></div></div></pre>

## Usage

To use the API, you can run the `main.py` file. The API receives two images as numpy arrays through a JSON object and sends whether the faces match or not through another JSON object.

## Methodology

The API uses an embedding-based methodology for face verification. This involves using a deep neural network to extract a fixed-length vector representation (embedding) for each face image. The embeddings are then compared using a distance metric (e.g. cosine similarity) to determine whether the two images belong to the same person or not.

ResNet50 is used as the neural network architecture to extract the embeddings. This is a widely used deep neural network architecture for image classification and feature extraction.

## Acknowledgements

This API was created using the following libraries:

* Google Cloud: for building the API
* Pillow: for image processing
* TensorFlow: for deep learning
* numpy: for numerical operations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
