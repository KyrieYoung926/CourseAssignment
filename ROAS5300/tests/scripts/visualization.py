import open3d as o3d
import numpy as np

class Visualizer:
    @staticmethod
    def create_point_cloud(points, colors):
        """创建点云对象"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        return pcd
    
    @staticmethod
    def create_camera_frame(pose, scale=1.0):
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
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])
        
        return line_set
    
    @staticmethod
    def show_reconstruction(points, colors, poses):
        """显示完整的重建结果"""
        # 创建点云
        pcd = Visualizer.create_point_cloud(points, colors)
        
        # 创建所有相机框架
        camera_frames = []
        for pose in poses:
            frame = Visualizer.create_camera_frame(pose)
            camera_frames.append(frame)
        
        # 显示结果
        o3d.visualization.draw_geometries([pcd] + camera_frames)