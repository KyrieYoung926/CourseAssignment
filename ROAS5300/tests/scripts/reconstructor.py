import numpy as np
import cv2
from pathlib import Path
from .feature_matcher import FeatureMatcher
from .geometry import Geometry
from .visualization import Visualizer

class SFMReconstructor:
    def __init__(self, img_dir, K):
        self.img_dir = Path(img_dir)
        self.K = K
        
        # 初始化组件
        self.matcher = FeatureMatcher()
        self.geometry = Geometry(K)
        self.visualizer = Visualizer()
        
        # 存储数据
        self.images = []
        self.image_names = []
        self.keypoints = []
        self.descriptors = []
        self.camera_poses = []
        self.points_3d = None
        self.point_colors = None
        self.point_2d_tracks = []
        self.registered_images = set()
        
        # 加载图像
        self.load_images()
    
    def load_images(self):
        """加载图像序列"""
        for i in range(11):  # 11张图片
            img_path = self.img_dir / f"{i}.png"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.images.append(img)
                    self.image_names.append(f"{i}.png")
        print(f"Loaded {len(self.images)} images")
    
    def extract_features(self):
        """从所有图像中提取特征"""
        for i, img in enumerate(self.images):
            kp, des = self.matcher.extract_features(img)
            self.keypoints.append(kp)
            self.descriptors.append(des)
            print(f"Image {self.image_names[i]}: {len(kp)} keypoints")
    
    def initialize_reconstruction(self):
        """使用前两张图片初始化重建"""
        # 获取特征匹配
        matches = self.matcher.match_features(self.descriptors[0], self.descriptors[1])
        pts1, pts2 = self.matcher.get_matched_points(self.keypoints[0], self.keypoints[1], matches)
        
        # 估计本质矩阵
        E, mask = self.geometry.estimate_essential_matrix(pts1, pts2)
        
        # 恢复相机位姿
        R, t, pose_mask = self.geometry.decompose_essential_matrix(E, pts1, pts2)
        
        # 设置相机位姿
        pose1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        pose2 = np.hstack((R, t))
        
        self.camera_poses = [pose1, pose2]
        self.registered_images.add(0)
        self.registered_images.add(1)
        
        # 三角化点
        valid_pts1 = pts1[mask.ravel() == 1]
        valid_pts2 = pts2[mask.ravel() == 1]
        points_3d = self.geometry.triangulate_points(valid_pts1, valid_pts2, pose1, pose2)
        
        # 获取颜色
        colors = self.get_colors_from_image(self.images[0], valid_pts1)
        
        self.points_3d = points_3d
        self.point_colors = colors
        
        # 创建特征轨迹
        for i in range(len(valid_pts1)):
            self.point_2d_tracks.append({0: valid_pts1[i], 1: valid_pts2[i]})
        
        return points_3d, colors
    
    def find_matches_between_images(self, idx1, idx2):
        """找到两张图像之间的匹配点"""
        matches = self.matcher.match_features(self.descriptors[idx1], self.descriptors[idx2])
        if len(matches) < 20:
            return None, None
        
        return self.matcher.get_matched_points(self.keypoints[idx1], self.keypoints[idx2], matches)
    
    def find_best_next_image(self):
        """找到最适合下一个处理的图像"""
        max_matches = 0
        best_image_idx = -1
        
        for idx in range(len(self.images)):
            if idx in self.registered_images:
                continue
                
            total_matches = 0
            for reg_idx in self.registered_images:
                pts1, pts2 = self.find_matches_between_images(reg_idx, idx)
                if pts1 is not None:
                    total_matches += len(pts1)
            
            if total_matches > max_matches:
                max_matches = total_matches
                best_image_idx = idx
        
        return best_image_idx
    
    def get_colors_from_image(self, image, points):
        """从图像中获取点的颜色"""
        colors = []
        for pt in points:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                colors.append(image[y, x])
            else:
                colors.append([0, 0, 0])
        return np.array(colors)
    
    def run_reconstruction(self):
        """执行完整的重建过程"""
        print("Extracting features...")
        self.extract_features()
        
        print("Initializing reconstruction...")
        self.initialize_reconstruction()
        
        print("Starting incremental reconstruction...")
        while len(self.registered_images) < len(self.images):
            next_idx = self.find_best_next_image()
            if next_idx == -1:
                break
                
            print(f"Processing image {next_idx}")
            success = self.process_next_image(next_idx)
            
            if not success:
                print(f"Failed to process image {next_idx}")
                continue
            
            # 定期进行bundle adjustment优化
            if len(self.registered_images) % 3 == 0:
                print("Running bundle adjustment...")
                self.geometry.bundle_adjustment(
                    self.points_3d,
                    self.camera_poses,
                    self.point_2d_tracks
                )
        
        # 最终的bundle adjustment优化
        print("Final bundle adjustment...")
        self.geometry.bundle_adjustment(
            self.points_3d,
            self.camera_poses,
            self.point_2d_tracks
        )
        
        print("Reconstruction complete!")

    def _get_colors_from_image(self, image: np.ndarray, points: np.ndarray):
        """从图像中获取点的颜色"""
        colors = []
        for pt in points:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                colors.append(image[y, x])
            else:
                colors.append([0, 0, 0])
        return np.array(colors)

    def process_next_image(self, image_idx: int):
        """处理下一张图像"""
        # 收集2D-3D对应关系
        points_2d = []
        points_3d = []
        
        for i, track in enumerate(self.point_2d_tracks):
            if any(idx in track for idx in self.registered_images):
                for reg_idx in self.registered_images:
                    if reg_idx in track:
                        points_2d.append(track[reg_idx])
                        points_3d.append(self.points_3d[i])
                        break
        
        if len(points_2d) < 10:
            return False
            
        # 估计新图像的位姿
        points_2d = np.array(points_2d)
        points_3d = np.array(points_3d)
        success, R, t = self.geometry.estimate_pnp_pose(points_3d, points_2d)
        print(f"Pose estimation success: {success}")
        if not success:
            return False
            
        # 保存新的相机位姿
        new_pose = np.hstack((R, t))
        self.camera_poses.append(new_pose)
        self.registered_images.add(image_idx)
        
        # 三角化新的点
        self._triangulate_new_points(image_idx)
        
        return True
    
    def save_reconstruction(self, output_dir):
        """
        保存重建结果
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 保存点云
        points_with_colors = np.hstack((self.points_3d, self.point_colors))
        np.savetxt(
            output_dir / 'points3d.txt',
            points_with_colors,
            header='X Y Z R G B',
            comments=''
        )
        
        # 保存相机位姿
        with open(output_dir / 'camera_poses.txt', 'w') as f:
            for i, pose in enumerate(self.camera_poses):
                R = pose[:3, :3]
                t = pose[:3, 3]
                f.write(f'Camera {i}:\n')
                f.write(f'R:\n{R}\n')
                f.write(f't:\n{t}\n\n')

    def visualize(self):
        """可视化重建结果"""
        self.visualizer.show_reconstruction(
            self.points_3d,
            self.point_colors,
            self.camera_poses
        )