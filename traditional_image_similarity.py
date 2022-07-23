import cv2
from skimage.metrics import structural_similarity
import numpy as np
import matplotlib.pyplot as plt
import imutils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.feature import peak_local_max
from skimage.feature import match_template
from sklearn.cluster import MeanShift, estimate_bandwidth
import time


major, minor, _ = cv2.__version__.split('.')
# pip install opencv-python==3.4.2 opencv-contrib-python==3.4.2


class ImageSimilarity(object):
    '''
    Class to implement similarity matching of two images
    '''
    def __init__(self, Hessian_Threshold:int = 400) -> None:
        self.sift = cv2.SIFT_create() 
        self.orb = cv2.ORB_create()
        self.surf = cv2.xfeatures2d.SURF_create(Hessian_Threshold) if int(major) <= 3 and int(minor) <= 4 else None # SURF is available in <= 3.4.2
        self.compare_methods = {"Correlation": cv2.HISTCMP_CORREL, "Chi-Squared": cv2.HISTCMP_CHISQR, "Intersection": cv2.HISTCMP_INTERSECT, "Hellinger": cv2.HISTCMP_BHATTACHARYYA}
        self. method_list = ["SIFT", "SURF", "ORB","SSIM","template_matching", "histogram_compare","template_match_histograms","scale_invariant_template_matching"]


    def compare_images(self, image_1:np.ndarray, image_2:np.ndarray, method:str = 'template_matching', **kwargs):
        '''
        Compare similarity based of different techniques
        image_1: Grayscale Image
        image2: grayscale second image
        method: List of techniques to use
        **kwargs: Keyword arguments for any of the algorithms
        '''
        assert len(image_1.shape) == len(image_2.shape) == 2, "both the images should be grayscale"

        if method == 'SIFT':
            return self.SIFT_SURF(image_1, image_2, method = "SIFT", **kwargs)

        elif method == "SURF":
            return self.SIFT_SURF(image_1, image_2, method = "SURF", **kwargs)
        
        elif method == "ORB":
            return self.SIFT_SURF(image_1, image_2, method = "ORB", **kwargs)

        elif method == "SSIM":
            return structural_similarity(image_1, image_2) # returns Mean structural similarity score

        elif method == "template_matching":
            return self.opencv_template_matching(image_1, image_2, **kwargs)

        elif method == "histogram_compare":
            return self.histogram_compare(image_1, image_2, **kwargs)[0]

        elif method == "template_match_histograms":
            return self.template_matching_on_histogram(image_1, image_2)

        else: raise NotImplementedError(f"Given method not found. Use one of {self.method_list}")


    def SIFT_SURF(self, img1:np.ndarray, img2:np.ndarray, method:str = "SIFT", use_128_dim:bool = False, match_threshold: float = 0.75, plot_image:bool = False)-> float:
        '''
        Get the descriptors from SIFT / SURF  / ORB and use FLANN to find the K Nearest matches
        https://datahacker.rs/feature-matching-methods-comparison-in-opencv/ : Comparisons
        https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html : SIFT
        https://docs.opencv.org/3.4/df/dd2/tutorial_py_surf_intro.html : SURF
        
        args:
            img1: First GRAYSCALE BGR image
            img2: Second GRAYSCALE BGR image
            MIN_MATCH_COUNT: Minimum number of equal features that have to be there
            method: ant of ['SURF', 'SIFT']
            use_128_dim: used in SURF to use the increased dimentionality
            match_threshold: Threshold that will be mulltiplied to the distance to find if a match is good or bad
            plot_image: Whether to plot image or not

        out:
            Will give you a % of "good" features matched out of all features
        '''
        if method == 'SURF':
            assert self.surf is not None, "SURF is available for version 3.4 or below"
            self.surf.setExtended(use_128_dim)
            kp1, des1 = self.surf.detectAndCompute(img1,None) 
            kp2, des2 = self.surf.detectAndCompute(img2,None)
        
        elif method == "SIFT":
            kp1, des1 = self.sift.detectAndCompute(img1,None) 
            kp2, des2 = self.sift.detectAndCompute(img2,None)

        else:
            kp1, des1 = self.orb.detectAndCompute(img1,None) 
            kp2, des2 = self.orb.detectAndCompute(img2,None)
        
        
        count = 0
        i = 1
        
        if method in ["SIFT", "SURF"]: # SIFT, SURF can use FLANN but ORB can't

            matches = self.FLANN(des1,des2) # get matches using FLANN
            count = 0 # count how many "Good" features we have out of all the features
            good_matches = [] # Need to draw only good matches, so create a mask
            for i,(m,n) in enumerate(matches): # ratio test as per Lowe's paper
                if m.distance < match_threshold*n.distance:
                    good_matches.append(m)
                    count += 1
        
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)
            good_matches = matches[:10]
        

        if plot_image:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

                if len(dst_pts) < 4 or len(src_pts) < 4: raise ValueError("At least 4 source and destination points are need to find Homography")

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
                matchesMask = mask.ravel().tolist()

                h,w = img1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                dst += (w, 0)

                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                                singlePointColor = None,
                                matchesMask = matchesMask, # draw only inliers
                                flags = 2)

                img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,None,**draw_params)
                img3 = cv2.polylines(img3, [np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)

                plt.imshow(img3)
                plt.show()

        return count / (i+1) # % of good features matched


    def FLANN(self, des1, des2, algorithm = 1, k = 2, trees = 5):
        '''
        Calculate FLANN: Fast Library for Approximate Nearest Neighbors scores based on descriptors
        Different algorithms like FLANN_INDEX_KDTREE = 1, FLANN_INDEX_LSH = 6 etc 

        Example: index_params= dict(algorithm = 6,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
        '''
        index_params = dict(algorithm = algorithm, trees = trees) # index_param are based on Algo used
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        return flann.knnMatch(des1,des2, k=k)


    def histogram_compare(self, image_1:np.ndarray, image_2:np.ndarray, compare_method:str = "Hellinger")-> tuple:
        '''
        Calculate and compare histograms of two images
        https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
        args:
            image1, image2: Numpy BGR images
            calculation_method: {"Correlation", "Chi-Squared", "Intersection","Hellinger"}  
        out:
            Return the matching score based on criteria along with the histograms
        '''
        #cv2.normalize(first_image_hist, first_image_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX) # can use normalization of both the histograms

        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])
        return cv2.compareHist(first_image_hist, second_image_hist, self.compare_methods[compare_method]), first_image_hist, second_image_hist


    def template_matching_on_histogram(self, image_1, image_2)->float:
        '''
        Create histogram of 2 images and then use them as a template matching
        '''
        img_hist_diff, first_image_hist, second_image_hist = self.histogram_compare(image_1, image_2)

        img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        # taking only 10% of histogram diff, since it's less accurate than template method
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff


    def skimage_template_matching(self, image, template, plot = True):
        '''
        Match template using skimage's template function
        '''
        sample_mt = match_template(image, template)
        patch_width, patch_height = template.shape

        bb = []
        for x, y in peak_local_max(sample_mt, threshold_abs=0.3):
            bb.append((y, x))

        if plot:
            fig, ax = plt.subplots(1,2,figsize=(20,10))
            ax[0].imshow(image,cmap='gray')
            ax[0].set_title('Original',fontsize=15)
            
            ax[1].imshow(image,cmap='gray')
            for y,x, in bb:
                rect = Rectangle((y, x), patch_height, patch_width, color='r', fc='none')
                ax[1].add_patch(rect)
            
            ax[1].set_title('Template Matched',fontsize=15)
            plt.show()
    
        return bb # list of all (x, y) Width and height are same as the template width and height

    
    def opencv_template_matching(self, image, template, plot_image:bool = True):
        '''
        Use Opencv's template matching function to find the template
        '''
        tH, tW = template.shape[:2]
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
        _, _, _, maxLoc = cv2.minMaxLoc(result)

        if plot_image:
            fig, ax = plt.subplots(1,2,figsize=(20,10))
            ax[0].imshow(image,cmap='gray')
            ax[0].set_title('Original',fontsize=15)
            
            ax[1].imshow(image, cmap = 'gray')
            rect = Rectangle((maxLoc[0],maxLoc[1], ), tW,tH, color='r', fc='none')
            ax[1].add_patch(rect)
            
            ax[1].set_title('Template Matched',fontsize=15)
            plt.show()

        return maxLoc # x, y Width and height are same as the template width and height

    
    def scale_invariant_template_matching(self, image, template, threshold:float = 0.6, show_live:bool = True):
        '''
        Idea is to pass in a template bigger in size and then keep on decreasi g its size so that we can match the template
        https://pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
        args:
            image: Original Query Image
            template: Template to find
            threshold: threshold for finding the boxes based on template matching
            show_live: whether to show how the BB are plotted
        '''
        img = image.copy()
        W, H = template.shape

        for scale in np.linspace(0.1, 0.99, 15)[::-1]: # Generate 15 points and resize the image according to the scale
            template = imutils.resize(template, height = int(H * scale))
            
            if template.shape[0] < 8 or template.shape[1] < 8: break # If template goes below 8 pixels, break

            w, h = template.shape[::-1]
            res = cv2.matchTemplate(image,template, cv2.TM_CCOEFF_NORMED)
            loc = np.where( res >= threshold)
            
            if show_live:
                img = image.copy()
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

                cv2.imshow("Visualization",img)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break

            time.sleep(2)

        cv2.destroyAllWindows()
   

def multi_obj_feature_matching(img1, img2, plot_live = False):
    '''
    Code Copied from: https://stackoverflow.com/questions/42938149/opencv-feature-matching-multiple-objects
    '''
    orb = cv2.ORB_create(10000, 1.2, nlevels=8, edgeThreshold = 5)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    x = np.array([kp2[0].pt])

    for i in range(len(kp2)):
        x = np.append(x, [kp2[i].pt], axis=0)

    x = x[1:len(x)]

    bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms.fit(x)
    labels = ms.labels_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)

    s = [None] * n_clusters_
    for i in range(n_clusters_):
        l = ms.labels_
        d, = np.where(l == i)
        print(d.__len__())
        s[i] = list(kp2[xx] for xx in d)

    des2_ = des2

    for i in range(n_clusters_):

        kp2 = s[i]
        l = ms.labels_
        d, = np.where(l == i)
        des2 = des2_[d, ]

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        des1 = np.float32(des1)
        des2 = np.float32(des2)

        matches = flann.knnMatch(des1, des2, 2)

    
        good = [] # store all the good matches as per Lowe's ratio test.
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>3:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)

            if M is None:
                print ("No Homography")
            elif plot_live:
                matchesMask = mask.ravel().tolist()

                h,w = img1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)

                img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                singlePointColor=None,
                                matchesMask=matchesMask,  # draw only inliers
                                flags=2)

                img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

                plt.imshow(img3, 'gray'), plt.show()

        else:
            print ("Not enough matches")




        





