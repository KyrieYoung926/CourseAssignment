import cv2
import numpy as np

class FeatureMatcher:
    def __init__(self):
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
    def extract_features(self, image):
        """提取图像特征点和描述子"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2, ratio_thresh=0.7):
        """使用比率测试进行特征匹配"""
        # 获取k近邻匹配
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # 应用比率测试
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                
        return good_matches
    
    @staticmethod
    def get_matched_points(keypoints1, keypoints2, matches):
        """获取匹配点的坐标"""
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        return pts1, pts2