# This projects different highlights different ways to recognise a product in the shelf

# Approaches:

There are two main ways of doing this. 
1. Using Traditional CV: Implemented 7 different methods. See  `traditional_image_similarity.py`
2. Using Deep Learning: Detectiing product using `Yolov5`


## Preprocessing: The given pdoduct image had floor in the background so:
```
1. Used `U2Net` to segment product only to reduce effect of noise.
2. Cropped the minimum boundign box.
3. Added White background to any space, if there is arounf the product.
4. Resized the image keeping aspect ration same so that images has the biggest dimension (any of height or width) as `480P`
```

## Traditional CV based Bounding Boxes
```
1. Find features from the images using methods in `traditional_image_similarity.py` suc has `SIFT, SURF, histogram etc`
2. Match the features (clusters), find features within a given threshold length`
3. Get Bounding boxes
```
![result_1](https://user-images.githubusercontent.com/50293852/180609777-2ffdcc38-0f1a-4f6c-b05c-d51ea829ae51.png)

Wasn't very effective. So I uesd deep Learning.

## Deep Learning based Bounding Box
```
1. Get Bounding Boxes from a trained `YOLOv5`
2. Crop each and every patch
```
![Detection](https://user-images.githubusercontent.com/50293852/180609949-67b3cbf8-4edc-4c56-a82b-765d434f6bac.jpeg)

## Image Similarity

Now you have 2 choices to choose from. Whether you use the aboce traditional methrods for feature matching or you use a pre trained model for image similarity.

1. **Traditional**: Get each patch and use any if the matching crireria from the above file `traditional_image_similarity.py`. It'll gice you a score in range `0-1` describing how similar the image patch is to the product given.

2. **Deep Learning** I used a `MobileNet` to get embeddings but the real problem is that the patches are so small. Some of the batches detected are of `8` pixels and `MiobileNet` dies not support less that `32`



