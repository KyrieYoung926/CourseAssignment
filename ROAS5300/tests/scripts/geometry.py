import numpy as np
import cv2
from scipy.optimize import least_squares

class Geometry:
    def __init__(self, K):
        self.K = K
        self.K_inv = np.linalg.inv(K)
    
    def estimate_essential_matrix(self, points1, points2):
        """估计本质矩阵"""
        # 归一化坐标
        norm_pts1 = self.normalize_points(points1)
        norm_pts2 = self.normalize_points(points2)
        
        E, mask = cv2.findEssentialMat(
            norm_pts1, norm_pts2,
            focal=1.0, pp=(0., 0.),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=3.0/self.K[0,0]
        )
        
        return E, mask
    
    def decompose_essential_matrix(self, E, points1, points2):
        """分解本质矩阵获取相机位姿"""
        # 归一化坐标
        norm_pts1 = self.normalize_points(points1)
        norm_pts2 = self.normalize_points(points2)
        
        _, R, t, mask = cv2.recoverPose(E, norm_pts1, norm_pts2)
        
        # 确保平移向量范数为1
        t = t / np.linalg.norm(t)
        
        return R, t, mask
    
    def normalize_points(self, points):
        """归一化图像坐标"""
        return np.dot(self.K_inv, np.vstack((points.T, np.ones(points.shape[0]))))[:2].T
    
    def triangulate_points(self, pts1, pts2, pose1, pose2):
        """三角化3D点"""
        P1 = np.dot(self.K, pose1)
        P2 = np.dot(self.K, pose2)
        
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def estimate_pnp_pose(self, points_3d, points_2d):
        """使用PnP估计相机位姿"""
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, self.K, None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec, inliers
    
    @staticmethod
    def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
        """创建BA稀疏矩阵结构"""
        m = len(point_indices)
        n = n_cameras * 6 + n_points * 3
        A = np.zeros((m * 2, n))
        
        i = np.arange(m)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1
            
        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
            
        return A