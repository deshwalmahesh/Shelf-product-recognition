# This project highlights different ways to recognise a product in the shelf
See `Product_detection.ipynb` for the whole end to end process.

# Approaches:

There are two main ways of doing this. 
1. Using Traditional CV: Implemented 8 different methods . See  `traditional_image_similarity.py`
```
[
'multi_obj_feature_matching', -> Create features and their clusters then find individual similarity between product and shelf image
'SIFT',
 'SURF',
 'ORB',
 'SSIM',  -> need images to be of same shape so not very helpful
 'template_matching', -> Uses two versions.  Scikit-image version uses "skimage.feature.peak_local_max" to filter noise
 'histogram_compare',
 'template_match_histograms', -> Find histogram of images and then use those as a template along with the original histogram difference criteria
 'scale_invariant_template_matching' -> Keep on decreasing the size of template until it mateches to all the products in the image. Lot of Manual tuning
 ]
 ```
 
2. Using Deep Learning: Detecting product using `Yolov5`


## Preprocessing: The given product image had floor in the background so:
```
1. Used `U2Net` to segment product only to reduce effect of noise.
2. Cropped the minimum boundign box.
3. Added White background to any space, if there is arounf the product.
4. Resized the image keeping aspect ration same so that images has the biggest dimension (any of height or width) as `480P`
```

## Finding Bounding Boxes 
### Traditional CV based approaches
```
1. Find features (clusters of features)  from the images using methods in `traditional_image_similarity.py` such has `SIFT, SURF etc`
2. Get Bounding boxes based on threshold.
```
![result_1](https://user-images.githubusercontent.com/50293852/180609777-2ffdcc38-0f1a-4f6c-b05c-d51ea829ae51.png)

Wasn't very effective. So I used deep Learning.

### Deep Learning based Bounding Box: `YOLOv5`
```
1. Get Bounding Boxes from a trained `YOLOv5`
2. Crop each and every patch
```
![Detection](https://user-images.githubusercontent.com/50293852/180609949-67b3cbf8-4edc-4c56-a82b-765d434f6bac.jpeg)


## Image Similarity

Now you have 2 choices to choose from. Whether you use the above traditional methrods for feature matching or you use a pre trained model for image similarity.

1. **Traditional**: Get each patch and use any of the matching crireria from the above file `traditional_image_similarity.py`. It'll give you a score in range `0-1` describing how similar the image patch is to the product given.

2. **Deep Learning** I used a `MobileNet` to get embeddings but the real problem is that the patches are so small. Some of the patches detected are of `8` pixels and `MobileNet` does not support less that `32`
