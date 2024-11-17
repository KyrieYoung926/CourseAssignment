import cv2
import numpy as np
from pathlib import Path
import numpy.linalg as LA
from scipy.optimize import least_squares
import open3d as o3d

class SFMReconstructor:
    def __init__(self, img_dir, intrinsic_matrix):
        """初始化SFM重建器"""
        self.img_dir = Path(img_dir)
        self.K = intrinsic_matrix
        self.K_inv = np.linalg.inv(intrinsic_matrix)
        
        # 基本属性
        self.images = []
        self.image_names = []
        self.keypoints = []
        self.descriptors = []
        
        # 重建结果
        self.camera_poses = []      # 存储所有相机位姿 [R|t]
        self.points_3d = []         # 存储三角化得到的3D点
        self.point_colors = []      # 存储3D点对应的颜色
        self.point_2d_tracks = []   # 存储3D点在各图像中的2D观测
        self.registered_images = set()  # 已注册的图像索引
        
        # 初始化特征提取和匹配器
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
        # 加载图像
        self.load_images()
    
    def find_matches_between_images(self, idx1, idx2):
        """找到两张图像之间的匹配点"""
        matches = self.match_features(idx1, idx2)
        if len(matches) < 20:  # 匹配点太少，认为匹配失败
            return None, None
        
        pts1 = np.float32([self.keypoints[idx1][m.queryIdx].pt for m in matches])
        pts2 = np.float32([self.keypoints[idx2][m.trainIdx].pt for m in matches])
        
        return pts1, pts2
    
    def find_best_next_image(self):
        """找到最适合下一个处理的图像"""
        max_matches = 0
        best_image_idx = -1
        
        for idx in range(len(self.images)):
            if idx in self.registered_images:
                continue
                
            # 计算与已注册图像的匹配点数
            total_matches = 0
            for reg_idx in self.registered_images:
                pts1, pts2 = self.find_matches_between_images(reg_idx, idx)
                if pts1 is not None:
                    total_matches += len(pts1)
            
            if total_matches > max_matches:
                max_matches = total_matches
                best_image_idx = idx
        
        return best_image_idx
    
    def triangulate_new_points(self, image_idx):
        """对新图像进行三角化"""
        new_points_3d = []
        new_colors = []
        new_2d_tracks = []
        
        # 遍历所有已注册的图像
        for ref_idx in self.registered_images:
            pts1, pts2 = self.find_matches_between_images(ref_idx, image_idx)
            if pts1 is None:
                continue
            
            # 三角化新的3D点
            pose1 = self.camera_poses[ref_idx]
            pose2 = self.camera_poses[image_idx]
            points_3d = self.triangulate_points(pts1, pts2, pose1, pose2)
            
            # 获取点的颜色
            colors = self.get_colors_from_image(self.images[ref_idx], pts1)
            
            # 创建观测轨迹
            for i in range(len(points_3d)):
                track = {
                    ref_idx: pts1[i],
                    image_idx: pts2[i]
                }
                new_2d_tracks.append(track)
            
            new_points_3d.extend(points_3d)
            new_colors.extend(colors)
        
        return np.array(new_points_3d), np.array(new_colors), new_2d_tracks
    
    def project_points(self, points_3d, pose, K):
        """投影3D点到图像平面"""
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # 转换到相机坐标系
        points_cam = np.dot(R, points_3d.T) + t.reshape(3, 1)
        
        # 投影到图像平面
        points_2d = np.dot(K, points_cam)
        points_2d = points_2d[:2] / points_2d[2]
        
        return points_2d.T
    
    def bundle_adjustment(self):
        """执行Bundle Adjustment优化"""
        
        def objective_function(x, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
            """BA的目标函数"""
            camera_params = x[:n_cameras * 6].reshape((n_cameras, 6))
            points_3d = x[n_cameras * 6:].reshape((n_points, 3))
            
            projected = np.zeros((len(points_2d), 2))
            
            for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
                pose = self.rodrigues_pose_to_matrix(camera_params[cam_idx])
                pt_proj = self.project_points(points_3d[pt_idx:pt_idx+1], pose, K)
                projected[i] = pt_proj[0]
            
            return (projected - points_2d).ravel()
        
        # 准备优化数据
        camera_indices = []
        point_indices = []
        points_2d = []
        
        for i, track in enumerate(self.point_2d_tracks):
            for img_idx, pt_2d in track.items():
                camera_indices.append(img_idx)
                point_indices.append(i)
                points_2d.append(pt_2d)
        
        camera_indices = np.array(camera_indices)
        point_indices = np.array(point_indices)
        points_2d = np.array(points_2d)
        
        # 初始参数
        n_cameras = len(self.camera_poses)
        n_points = len(self.points_3d)
        
        camera_params = []
        for pose in self.camera_poses:
            R = pose[:3, :3]
            t = pose[:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            camera_params.extend(rvec.ravel())
            camera_params.extend(t.ravel())
        
        x0 = np.concatenate([camera_params, self.points_3d.ravel()])
        print(f"x0 shape: {x0.shape}")
        # 执行优化
        print(f"Number of cameras: {n_cameras}")
        print(f"Number of points: {n_points}")
        print(f"Number of 2D points: {len(points_2d)}")
        print(f"Residual count: {len(points_2d) * 2}")
        print(f"Variable count: {n_cameras * 6 + n_points * 3}")

        res = least_squares(
            objective_function, x0,
            args=(n_cameras, n_points, camera_indices, point_indices, points_2d, self.K),
            method='trf',
            max_nfev=100
        )
        
        # 更新结果
        camera_params = res.x[:n_cameras * 6].reshape((n_cameras, 6))
        self.points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))
        
        # 更新相机位姿
        for i, params in enumerate(camera_params):
            R, _ = cv2.Rodrigues(params[:3])
            t = params[3:]
            self.camera_poses[i] = np.hstack((R, t.reshape(3, 1)))

    def estimate_pnp_pose(self, points_3d, points_2d):
        """
        使用PnP估计新图像的相机位姿
        """
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, self.K, None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # 转换旋转向量为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        return R, tvec, inliers
    
    def incremental_reconstruction(self):
        """执行增量式重建"""
        # 初始重建
        self.initial_reconstruction()
        self.registered_images.add(0)
        self.registered_images.add(1)
        
        while len(self.registered_images) < len(self.images):
            # 找到最佳的下一张图像
            next_idx = self.find_best_next_image()
            if next_idx == -1:
                break
                
            print(f"Processing image {next_idx}")
            
            # 使用PnP估计新图像的位姿
            points_2d = []
            points_3d = []
            print(f"Point 2D tracks: {self.point_2d_tracks}")
            print(f"Number of tracks: {len(self.point_2d_tracks)}")

            for i, track in enumerate(self.point_2d_tracks):
                if any(idx in track for idx in self.registered_images):
                    for reg_idx in self.registered_images:
                        if reg_idx in track:
                            points_2d.append(track[reg_idx])
                            points_3d.append(self.points_3d[i])
                            break
            
            if len(points_2d) < 10:
                continue
                
            R, t, inliers = self.estimate_pnp_pose(
                np.array(points_3d),
                np.array(points_2d)
            )
            
            # 保存新的相机位姿
            new_pose = np.hstack((R, t))
            self.camera_poses.append(new_pose)
            print(f"Adding image {next_idx} to registered images")
            self.registered_images.add(next_idx)
            
            # 三角化新的点
            new_points, new_colors, new_tracks = self.triangulate_new_points(next_idx)
            
            # 更新重建结果
            if len(new_points) > 0:
                self.points_3d = np.vstack((self.points_3d, new_points))
                self.point_colors = np.vstack((self.point_colors, new_colors))
                self.point_2d_tracks.extend(new_tracks)
            
            # 定期执行BA优化
            if len(self.registered_images) % 3 == 0:
                self.bundle_adjustment()
        
        # 最终的BA优化
        self.bundle_adjustment()
    
    def visualize_reconstruction(self):
        """可视化重建结果"""
        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points_3d)
        pcd.colors = o3d.utility.Vector3dVector(self.point_colors / 255.0)
        
        # 创建相机框架
        camera_frames = []
        for pose in self.camera_poses:
            frame = self.create_camera_frame(pose)
            camera_frames.extend(frame)
        
        # 可视化
        o3d.visualization.draw_geometries([pcd] + camera_frames)
    
    def create_camera_frame(self, pose, scale=1.0):
        """创建相机框架的可视化"""
        points = np.array([
            [0, 0, 0],
            [1, 1, 2],
            [-1, 1, 2],
            [1, -1, 2],
            [-1, -1, 2]
        ]) * scale
        
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 4], [4, 3], [3, 1]
        ]
        
        # 转换到世界坐标系
        R = pose[:3, :3]
        t = pose[:3, 3]
        points = (R @ points.T + t.reshape(3, 1)).T
        
        # 创建线框
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])
        
        return [line_set]
    
    @staticmethod
    def rodrigues_pose_to_matrix(params):
        """将Rodriguez旋转向量和平移向量转换为变换矩阵"""
        R, _ = cv2.Rodrigues(params[:3])
        t = params[3:]
        return np.hstack((R, t.reshape(3, 1)))
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
    
    def initial_reconstruction(self):
        """
        使用前两张图片进行初始重建
        """
        # 获取特征匹配
        matches = self.match_features(0, 1)
        
        # 获取匹配点坐标
        pts1 = np.float32([self.keypoints[0][m.queryIdx].pt for m in matches])
        pts2 = np.float32([self.keypoints[1][m.trainIdx].pt for m in matches])
        
        # 计算本质矩阵
        E, mask = self.estimate_essential_matrix(pts1, pts2)
        
        # 分解本质矩阵得到位姿
        R, t, pose_mask = self.decompose_essential_matrix(E, pts1, pts2)
        
        # 设置第一个相机作为世界坐标系
        pose1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # [I|0]
        pose2 = np.hstack((R, t))  # [R|t]
        
        # 保存相机位姿
        self.camera_poses = [pose1, pose2]
        
        # 三角化特征点
        valid_pts1 = pts1[mask.ravel() == 1]
        valid_pts2 = pts2[mask.ravel() == 1]
        points_3d = self.triangulate_points(valid_pts1, valid_pts2, pose1, pose2)
        
        # 获取点的颜色
        colors = self.get_colors_from_image(self.images[0], valid_pts1)
        
        # 保存重建结果
        self.points_3d = points_3d
        self.point_colors = colors

        # **新增代码：初始化 2D 点轨迹**
        self.point_2d_tracks = []
        for i in range(len(points_3d)):
            self.point_2d_tracks.append({
                0: tuple(valid_pts1[i]),  # 第0张图像的2D点
                1: tuple(valid_pts2[i]),  # 第1张图像的2D点
            })
        
        print("Initial reconstruction completed:")
        print(f"Points 3D: {self.points_3d.shape[0]}")
        print(f"Point 2D Tracks: {len(self.point_2d_tracks)}")

        return points_3d, colors

    def estimate_essential_matrix(self, points1, points2):
        """
        估计本质矩阵
        """
        # 归一化坐标
        norm_pts1 = self.normalize_points(points1)
        norm_pts2 = self.normalize_points(points2)
        
        # 使用RANSAC估计本质矩阵
        E, mask = cv2.findEssentialMat(
            norm_pts1, norm_pts2, 
            focal=1.0, pp=(0., 0.),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=3.0/self.K[0,0]
        )
        
        return E, mask
    
    def decompose_essential_matrix(self, E, points1, points2):
        """
        分解本质矩阵得到相机位姿
        """
        # 归一化坐标
        norm_pts1 = self.normalize_points(points1)
        norm_pts2 = self.normalize_points(points2)
        
        # 恢复相对位姿
        _, R, t, mask = cv2.recoverPose(E, norm_pts1, norm_pts2)
        
        # 确保平移向量的范数为1
        t = t / np.linalg.norm(t)
        
        return R, t, mask
    
    def triangulate_points(self, pts1, pts2, pose1, pose2):
        """
        三角化得到3D点
        """
        P1 = np.dot(self.K, pose1)  # 3x4 projection matrix for first camera
        P2 = np.dot(self.K, pose2)  # 3x4 projection matrix for second camera
        
        # 三角化
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # 转换为3D点
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def get_colors_from_image(self, image, points):
        """
        从图像中获取点的颜色
        """
        colors = []
        for pt in points:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                colors.append(image[y, x])
            else:
                colors.append([0, 0, 0])
        return np.array(colors)
    def normalize_points(self, points):
        """
        归一化图像坐标
        """
        return np.dot(self.K_inv, np.vstack((points.T, np.ones(points.shape[0]))))[:2].T
    
# 主程序
if __name__ == "__main__":
    # 相机内参（需要替换为实际值）
    K = np.array([
        [1000, 0, 512],
        [0, 1000, 341],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 初始化重建器
    reconstructor = SFMReconstructor("ROAS5300/datasets/Foutain_Comp", K)
    
    # 提取特征
    reconstructor.extract_features()
    
    # 执行重建
    reconstructor.incremental_reconstruction()
    
    # 保存结果
    reconstructor.save_reconstruction("reconstruction_results")
    
    # 可视化
    reconstructor.visualize_reconstruction()