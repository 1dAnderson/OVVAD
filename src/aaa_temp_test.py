import numpy as np

# 指定文件路径
file_path = "list/ucf/ucf-prompt_1_not16_859.npy"

# 读取 .npy 文件
data = np.load(file_path, allow_pickle=True)

# 输出形状
print("Shape of the loaded array:", data.shape)
print(data)