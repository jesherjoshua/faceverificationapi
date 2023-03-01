<h1 align=center>Face API</h1>
<p align=center>provide 2 faces and return 'Yes' or 'No' if it matches or not</p>
<div align='center'>
  
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/jesherjoshua/faceverificationapi?logo=github&style=for-the-badge)](https://github.com/jesherjoshua/faceverificationapi)
[![Languages](https://img.shields.io/github/languages/count/jesherjoshua/faceverificationapi?style=for-the-badge)](https://github.com/jesherjoshua/faceverificationapi)
[![GitHub last commit](https://img.shields.io/github/last-commit/jesherjoshua/faceverificationapi?style=for-the-badge&logo=git)](https://github.com/jesherjoshua/faceverificationapi)

<h2>Built With</h2>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)
</div>


This is a Python-based API for face verification using an embedding-based methodology and ResNet50 to create the embeddings. The API consists of two main files: `main.py` and `requirements.txt`.

## Installation

To install the required dependencies, run the following command:

<pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg" data-darkreader-inline-stroke=""><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg></button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs">pip install -r requirements.txt
</code></div></div></pre>

## Usage

To use the API, you can run the `main.py` file. The API receives two images as numpy arrays through a JSON object and sends whether the faces match or not through another JSON object.

## Methodology

The API uses an embedding-based methodology for face verification. This involves using a deep neural network to extract a fixed-length vector representation (embedding) for each face image. The embeddings are then compared using a distance metric (e.g. cosine similarity) to determine whether the two images belong to the same person or not.

ResNet50 is used as the neural network architecture to extract the embeddings. This is a widely used deep neural network architecture for image classification and feature extraction.

## Acknowledgements

This API was created using the following libraries:

* Google Cloud: for building the API
* OpenCV: for image processing
* TensorFlow: for deep learning
* numpy: for numerical operations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
