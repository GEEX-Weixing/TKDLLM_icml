import matplotlib.pyplot as plt

# 读取文件中的数值
filename = 'data.txt'  # 替换为你的文件名
data = []
with open(filename, 'r') as file:
    for line in file:
        # 去除每行的换行符并转换为浮点数
        data.append(float(line.strip()))

# 绘制折线图
plt.plot(data, marker='o', linestyle='-')  # 'o'标记每个点，'-'表示线
plt.title("Line Plot of Data from File")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.show()
