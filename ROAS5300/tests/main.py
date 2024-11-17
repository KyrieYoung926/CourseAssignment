from scripts.reconstructor import SFMReconstructor
import numpy as np

def main():
    # 相机内参矩阵
    K = np.array([
        [1000, 0, 512],
        [0, 1000, 341],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 初始化重建器
    reconstructor = SFMReconstructor("ROAS5300/datasets/Foutain_Comp", K)
    
    # 执行重建
    reconstructor.run_reconstruction()
    
    # 保存结果
    reconstructor.save_reconstruction("reconstruction_results")
    
    # 可视化
    reconstructor.visualize()

if __name__ == "__main__":
    main()