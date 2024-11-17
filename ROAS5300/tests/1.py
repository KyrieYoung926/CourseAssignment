import cv2
import numpy as np
from pathlib import Path
import os

class SFMReconstructor:
    def __init__(self, img_dir, intrinsic_matrix):
        """
        初始化SFM重建器
        Args:
            img_dir: 图像目录路径
            intrinsic_matrix: 相机内参矩阵
        """
        self.img_dir = Path(img_dir)
        self.K = intrinsic_matrix
        self.images = []
        self.image_names = []
        self.load_images()
        
        # 特征提取器和匹配器
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        # 存储特征点和描述子
        self.keypoints = []     # 每张图的关键点
        self.descriptors = []   # 每张图的描述子
        self.matches_list = []  # 图像间的匹配
        
    def load_images(self):
        """加载图像"""
        for i in range(11):  # 11张图片
            img_path = self.img_dir / f"{i}.png"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.images.append(img)
                    self.image_names.append(f"{i}.png")
        print(f"Loaded {len(self.images)} images")
    
    def extract_features(self):
        """从所有图像中提取SIFT特征"""
        for i, img in enumerate(self.images):
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 提取特征
            kp, des = self.feature_detector.detectAndCompute(gray, None)
            
            self.keypoints.append(kp)
            self.descriptors.append(des)
            print(f"Image {self.image_names[i]}: {len(kp)} keypoints")
    
    def match_features(self, idx1, idx2, ratio_thresh=0.7):
        """
        匹配两张图像的特征点
        Args:
            idx1, idx2: 要匹配的两张图片的索引
            ratio_thresh: Lowe's ratio test阈值
        Returns:
            matches: 筛选后的匹配
        """
        # 使用k最近邻匹配
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(self.descriptors[idx1], self.descriptors[idx2], k=2)
        
        # 应用Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def draw_matches(self, idx1, idx2, matches):
        """
        绘制两张图像间的匹配结果
        """
        img_matches = cv2.drawMatches(
            self.images[idx1], self.keypoints[idx1],
            self.images[idx2], self.keypoints[idx2],
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return img_matches

# 示例使用
if __name__ == "__main__":
    # 相机内参矩阵 (这里需要填入实际的内参矩阵)
    K = np.array([
        [1000, 0, 512],  # 这里使用示例值，需要替换为实际值
        [0, 1000, 341],
        [0, 0, 1]
    ])
    
    # 初始化重建器
    img_dir = "ROAS5300/datasets/Foutain_Comp"
    reconstructor = SFMReconstructor(img_dir, K)
    
    # 提取特征
    reconstructor.extract_features()
    
    # 测试第一张和第二张图片的匹配
    matches = reconstructor.match_features(0, 1)
    print(f"Found {len(matches)} good matches between image 0 and 1")
    
    # 显示匹配结果
    match_img = reconstructor.draw_matches(0, 1, matches)
    cv2.imshow("Matches", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()