import cv2
import numpy as np
import os
from typing import List, Tuple, Dict
import open3d as o3d

class ImageData:
    def __init__(self, image: np.ndarray, K: np.ndarray):
        self.image = image
        self.K = K  # 内参矩阵
        self.R = None  # 旋转矩阵
        self.t = None  # 平移向量
        self.keypoints = None
        self.descriptors = None

class SFMReconstructor:
    def __init__(self):
        self.images: List[ImageData] = []
        self.points3D = []
        self.colors = []
        self.point_tracks = {}  # 添加这行
        self.next_point_id = 0  # 添加这行
        
    def load_images(self, image_dir: str, K: np.ndarray):
        """加载图像并存储内参"""
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            self.images.append(ImageData(image, K))
    
    def extract_features(self):
        """使用SIFT提取特征点和描述子"""
        sift = cv2.SIFT_create()
        for img_data in self.images:
            gray = cv2.cvtColor(img_data.image, cv2.COLOR_BGR2GRAY)
            img_data.keypoints, img_data.descriptors = sift.detectAndCompute(gray, None)
    
    def match_features(self, img1: ImageData, img2: ImageData) -> List[Tuple[int, int]]:
        """匹配两张图像的特征点"""
        # FLANN参数
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        # 创建FLANN匹配器
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 使用knnMatch进行特征匹配
        matches = flann.knnMatch(img1.descriptors, img2.descriptors, k=2)
        
        # 应用比率测试
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # 可以调整这个阈值
                good_matches.append((m.queryIdx, m.trainIdx))
        
        print(f"Found {len(good_matches)} good matches between images")
        return good_matches

    def estimate_essential_matrix(self, img1: ImageData, img2: ImageData, 
                                matches: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """估计本质矩阵并恢复相对位姿"""
        # 获取匹配点的坐标
        pts1 = np.float32([img1.keypoints[m[0]].pt for m in matches])
        pts2 = np.float32([img2.keypoints[m[1]].pt for m in matches])
        
        # 将像素坐标转换为归一化坐标
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), img1.K, None)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), img2.K, None)
        
        # 计算本质矩阵
        E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, pp=(0., 0.),
                                      method=cv2.RANSAC, prob=0.999, threshold=0.001)
        
        # 从本质矩阵恢复相对位姿
        _, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm)
        
        return E, R, t, mask.ravel().astype(bool)
    
    def triangulate_points(self, img1: ImageData, img2: ImageData, 
                         matches: List[Tuple[int, int]], mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """三角化重建3D点"""
        # 获取匹配点
        pts1 = np.float32([img1.keypoints[m[0]].pt for m in matches])[mask]
        pts2 = np.float32([img2.keypoints[m[1]].pt for m in matches])[mask]
        
        # 构建投影矩阵
        P1 = img1.K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # 第一个相机 [R|t] = [I|0]
        P2 = img2.K @ np.hstack((img2.R, img2.t))               # 第二个相机 [R|t]
        
        # 三角化
        points4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points3D = (points4D / points4D[3]).T[:, :3]
        
        # 获取对应的颜色信息
        colors = np.array([img1.image[int(pt[1]), int(pt[0])] for pt in pts1])
        
        return points3D, colors
    
    def initialize_structure(self):
        """使用前两张图片初始化结构"""
        if len(self.images) < 2:
            raise ValueError("Need at least two images for initialization")
            
        # 设置第一个相机的位姿
        self.images[0].R = np.eye(3)
        self.images[0].t = np.zeros((3, 1))
        
        # 匹配特征点
        matches = self.match_features(self.images[0], self.images[1])
        
        # 估计本质矩阵和相对位姿
        _, R, t, mask = self.estimate_essential_matrix(self.images[0], self.images[1], matches)
        
        # 设置第二个相机的位姿
        self.images[1].R = R
        self.images[1].t = t
        
        # 只使用内点进行三角化
        valid_matches = [m for i, m in enumerate(matches) if mask[i]]
        
        # 三角化重建3D点
        points3D, colors = self.triangulate_points(self.images[0], self.images[1], 
                                                valid_matches, np.ones(len(valid_matches), dtype=bool))
        
        # 存储重建结果
        self.points3D = points3D.tolist()  # 转换为list便于后续添加新点
        self.colors = colors.tolist()
        
        print(f"Found {len(matches)} matches, {len(valid_matches)} valid matches after RANSAC")
        print(f"Triangulated {len(self.points3D)} 3D points")
        
        return len(points3D)
    
    def estimate_pose_pnp(self, img_data: ImageData, points3D: np.ndarray, 
                         points2D: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """使用PnP估计相机位姿"""
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points3D, points2D, img_data.K, None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=0.99
        )
        
        if not success:
            return False, None, None
            
        R, _ = cv2.Rodrigues(rvec)
        return True, R, tvec
    
    def triangulate_new_points(self, img_data: ImageData, ref_img: ImageData, 
                             matches: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
        """三角化新的特征点对"""
        pts1 = np.float32([ref_img.keypoints[m[0]].pt for m in matches])
        pts2 = np.float32([img_data.keypoints[m[1]].pt for m in matches])
        
        P1 = ref_img.K @ np.hstack((ref_img.R, ref_img.t))
        P2 = img_data.K @ np.hstack((img_data.R, img_data.t))
        
        points4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points3D = (points4D / points4D[3]).T[:, :3]
        
        # 检查三角化点的有效性
        valid = []
        for pt3D, pt2D in zip(points3D, pts2):
            # 检查重投影误差
            projected = img_data.K @ (img_data.R @ pt3D.reshape(3,1) + img_data.t)
            projected = projected[:2] / projected[2]
            error = np.linalg.norm(projected.ravel() - pt2D)
            
            # 检查深度是否为正
            P2_center = -np.linalg.inv(img_data.R) @ img_data.t
            vec = pt3D - P2_center.ravel()
            depth = vec @ img_data.R[2]
            
            valid.append(error < 5.0 and depth > 0)
        
        valid = np.array(valid)
        colors = np.array([ref_img.image[int(pt[1]), int(pt[0])] for pt in pts1[valid]])
        
        return points3D[valid], colors, valid
    
    def add_new_image(self, img_idx: int) -> bool:
        """增量式添加新图像"""
        new_img = self.images[img_idx]
        best_num_inliers = 0
        best_R = None
        best_t = None
        
        # 寻找最佳参考图像
        for ref_idx in range(img_idx):
            ref_img = self.images[ref_idx]
            if ref_img.R is None:  # 跳过未成功重建的图像
                continue
                
            matches = self.match_features(ref_img, new_img)
            if len(matches) < 30:  # 匹配点太少
                continue
            
            # 找到已经重建的3D点
            points3D = []
            points2D = []
            for m in matches:
                for track_id, projections in self.point_tracks.items():
                    if (ref_idx, m[0]) in projections:
                        point3D_idx = track_id
                        points3D.append(self.points3D[point3D_idx])
                        points2D.append(new_img.keypoints[m[1]].pt)
            
            if len(points3D) < 10:  # 3D-2D对应点太少
                continue
                
            points3D = np.array(points3D)
            points2D = np.array(points2D)
            
            # PnP估计位姿
            success, R, t = self.estimate_pose_pnp(new_img, points3D, points2D)
            if not success:
                continue
                
            if len(points2D) > best_num_inliers:
                best_num_inliers = len(points2D)
                best_R = R
                best_t = t
        
        if best_R is None:
            return False
            
        # 设置新图像的位姿
        new_img.R = best_R
        new_img.t = best_t
        
        # 三角化新的3D点
        for ref_idx in range(img_idx):
            ref_img = self.images[ref_idx]
            if ref_img.R is None:
                continue
                
            matches = self.match_features(ref_img, new_img)
            new_points3D, new_colors, valid = self.triangulate_new_points(new_img, ref_img, matches)
            
            # 更新重建结果
            for i, (is_valid, pt3D, color) in enumerate(zip(valid, new_points3D, new_colors)):
                if is_valid:
                    self.points3D.append(pt3D)
                    self.colors.append(color)
                    match = matches[i]
                    self.point_tracks[self.next_point_id] = [(ref_idx, match[0]), (img_idx, match[1])]
                    self.next_point_id += 1
        
        return True
    
    def incremental_reconstruction(self):
        """增量式重建所有图像"""
        # 初始化前两张图像
        num_initial = self.initialize_structure()
        print(f"Initial reconstruction: {num_initial} points")
        
        # 获取前两张图像之间的匹配，并只保留成功三角化的匹配
        matches_01 = self.match_features(self.images[0], self.images[1])
        _, _, _, mask = self.estimate_essential_matrix(self.images[0], self.images[1], matches_01)
        valid_matches = [m for i, m in enumerate(matches_01) if mask[i]]
        
        # 初始化点追踪信息
        for i in range(len(self.points3D)):
            if i < len(valid_matches):  # 确保不会越界
                self.point_tracks[i] = [(0, valid_matches[i][0]), (1, valid_matches[i][1])]
        self.next_point_id = len(self.points3D)
        
        # 增量式添加剩余图像
        for i in range(2, len(self.images)):
            success = self.add_new_image(i)
            if success:
                print(f"Added image {i}, total points: {len(self.points3D)}")
            else:
                print(f"Failed to add image {i}")
    
    def get_camera_positions(self) -> np.ndarray:
        """获取所有相机中心位置"""
        camera_centers = []
        for img in self.images:
            if img.R is not None:
                center = -img.R.T @ img.t
                camera_centers.append(center.ravel())
        return np.array(camera_centers)
    
def visualize_point_cloud(ply_path):
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 添加点云到可视化器
    vis.add_geometry(pcd)
    
    # 设置视角和渲染选项
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0, 0, 0])  # 黑色背景
    
    # 运行可视化器
    vis.run()
    vis.destroy_window()

def main():
    # 示例内参矩阵 (根据实际数据调整)
    K = np.array([
        [800, 0, 512],
        [0, 800, 341],
        [0, 0, 1]
    ],dtype=np.float32)
    
    reconstructor = SFMReconstructor()
    
    # 假设图片在./images目录下
    image_dir = "/home/xunyang/Desktop/Projects/CourseAssignment/ROAS5300/datasets/Foutain_Comp"
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Directory {image_dir} does not exist")
        
    reconstructor.load_images(image_dir, K)
    reconstructor.extract_features()
    
    # 初始化结构
    num_points = reconstructor.initialize_structure()
    print(f"初始重建得到 {num_points} 个3D点")
    
    # 执行增量式重建
    reconstructor.incremental_reconstruction()
    
    # 保存点云
    def save_point_cloud(points3D, colors, filename):
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points3D)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for pt, color in zip(points3D, colors):
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {color[2]} {color[1]} {color[0]}\n")
    
    save_point_cloud(reconstructor.points3D, reconstructor.colors, "initial_reconstruction.ply")
    visualize_point_cloud("initial_reconstruction.ply")
if __name__ == "__main__":
    main()