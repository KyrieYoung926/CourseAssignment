import cv2
import numpy as np
from pathlib import Path
import json

def load_images(img_dir):
    """
    加载图像目录中的所有图像
    
    Args:
        img_dir: 图像目录路径
    Returns:
        images: 图像列表
        image_names: 图像名称列表
    """
    img_dir = Path(img_dir)
    images = []
    image_names = []
    
    # 支持多种图像格式
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # 按文件名排序加载图像
    for img_path in sorted(img_dir.glob('*')):
        if img_path.suffix.lower() in valid_extensions:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                image_names.append(img_path.name)
    
    if not images:
        raise ValueError(f"No valid images found in {img_dir}")
        
    print(f"Loaded {len(images)} images from {img_dir}")
    return images, image_names

def save_points_cloud(output_dir, points_3d, point_colors):
    """
    保存点云数据为PLY格式
    
    Args:
        output_dir: 输出目录
        points_3d: Nx3数组，包含3D点坐标
        point_colors: Nx3数组，包含对应的RGB颜色
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为PLY格式
    with open(output_dir / 'points3d.ply', 'w') as f:
        # 写入PLY头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入点云数据
        for point, color in zip(points_3d, point_colors):
            f.write(f"{point[0]} {point[1]} {point[2]} ")
            f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
    
    # 同时保存为TXT格式便于查看
    points_with_colors = np.hstack((points_3d, point_colors))
    np.savetxt(
        output_dir / 'points3d.txt',
        points_with_colors,
        header='X Y Z R G B',
        fmt='%.6f %.6f %.6f %d %d %d'
    )

def save_camera_poses(output_dir, camera_poses):
    """
    保存相机位姿
    
    Args:
        output_dir: 输出目录
        camera_poses: 相机位姿列表，每个位姿是3x4矩阵[R|t]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON格式
    poses_data = []
    for i, pose in enumerate(camera_poses):
        R = pose[:3, :3].tolist()
        t = pose[:3, 3].tolist()
        poses_data.append({
            'camera_id': i,
            'R': R,
            't': t
        })
    
    with open(output_dir / 'camera_poses.json', 'w') as f:
        json.dump(poses_data, f, indent=2)
    
    # 同时保存为可读性更好的TXT格式
    with open(output_dir / 'camera_poses.txt', 'w') as f:
        for i, pose in enumerate(camera_poses):
            R = pose[:3, :3]
            t = pose[:3, 3]
            f.write(f'Camera {i}:\n')
            f.write('R:\n')
            for row in R:
                f.write(f'{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n')
            f.write('t:\n')
            f.write(f'{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n\n')
