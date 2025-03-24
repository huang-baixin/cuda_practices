import matplotlib.pyplot as plt

# 解析数据
data = """
gpu Neighbored  elapsed 0.026080 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Neighbored2 elapsed 0.007243 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Interleaved elapsed 0.006216 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Unrolling2  elapsed 0.003465 sec gpu_sum: 2139353471 <<<grid 16384 block 512>>>
gpu Unrolling4  elapsed 0.001911 sec gpu_sum: 2139353471 <<<grid 8192 block 512>>>
gpu Unrolling8  elapsed 0.002759 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu UnrollWarp8 elapsed 0.002803 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll8  elapsed 0.002798 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
gpu Cmptnroll   elapsed 0.002798 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
"""

# 提取时间数据
lines = data.strip().split('\n')
labels = []
times = []

for line in lines:
    parts = line.split()
    if len(parts) >= 4:
        labels.append(parts[1])
        times.append(float(parts[3]))

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(labels, times, color='blue', label='Execution Time')

# 添加标题和标签
plt.title('Execution Times of Different CUDA Kernels')
plt.xlabel('Kernel Name')
plt.ylabel('Time (sec)')
plt.xticks(rotation=45)
plt.legend()

# 保存图像
plt.tight_layout()
plt.savefig('cuda_kernels_times.png', dpi=300)

# 显示图形（可选）
plt.show()