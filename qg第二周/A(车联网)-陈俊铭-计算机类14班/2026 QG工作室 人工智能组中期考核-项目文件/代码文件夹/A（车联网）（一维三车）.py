import numpy as np
import matplotlib.pyplot as plt

# 定义变量
N = 3                     # 车辆数（索引0是领导，1和2是跟随）
dt = 0.01                 # 步长
T = 20                    # 时间
steps = int(T/dt)         # 步数
beta = 1.0                # 邻居速度差增益
gamma = 1.0               # 领导速度差增益

# 邻接矩阵（全连通，无向）
# A = np.array([    # 断联
#     [0, 0, 0],
#     [0, 0, 1],
#     [0, 1, 0]
# ])
A = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

# 领导连接标志：领导自己为0，跟随车都能收到领导信息
k = np.array([0, 1, 1])

# 期望间距 r_ij（一维，标量矩阵），定义跟随车相对于领导的目标位置
# 设领导在0，跟随1在后方2米，跟随2在后方4米（期望位置分别为 -2, -4）
target_pos = np.array([0, -2, -4])   # 期望位置（相对于领导）
# 计算 r_ij = target_i - target_j   (满足反对称)
r_ij = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        r_ij[i, j] = target_pos[i] - target_pos[j]

# 期望与领导间距 r_i = target_i - target_0，实际上就是 target_pos
r_i = target_pos.copy()

# 初始状态：领导在0，速度1；跟随在正前方较远处，速度0
x = np.array([0.0, 5.0, 8.0])    # 位置
v = np.array([1.0, 0.0, 0.0])    # 速度

# 领导控制输入（匀速）
u_leader = 0.0

# 记录历史
x_hist = [x.copy()]
v_hist = [v.copy()]

# 仿真循环
for step in range(steps):
    u = np.zeros(N)
    # 对每辆车计算控制输入
    for i in range(N):
        # 邻居项
        neighbor_sum = 0.0
        for j in range(N):
            if A[i, j] != 0:
                pos_err = (x[i] - x[j] - r_ij[i, j])
                vel_err = beta * (v[i] - v[j])
                neighbor_sum += pos_err + vel_err
        u[i] -= neighbor_sum   # 负号

        # 领导项
        if k[i] != 0:
            pos_err_leader = (x[i] - x[0] - r_i[i])
            vel_err_leader = gamma * (v[i] - v[0])
            u[i] -= k[i] * (pos_err_leader + vel_err_leader)

    # 单独设定领导控制输入
    u[0] = u_leader  # 强制覆盖领导车的值

    # 欧拉更新
    x = x + v * dt
    v = v + u * dt

    # 记录
    x_hist.append(x.copy())
    v_hist.append(v.copy())

# 转换为数组并生成时间轴
x_hist = np.array(x_hist)
v_hist = np.array(v_hist)
t = np.linspace(0, T, steps+1)

# ---------------------------------------------------------------------------------------

# 绘图
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
for i in range(N):
    plt.plot(t, x_hist[:, i], label=f'vehicle {i}')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.legend()
plt.title('Position')

plt.subplot(1,3,2)
for i in range(N):
    plt.plot(t, v_hist[:, i], label=f'vehicle {i}')
plt.xlabel('time (s)')
plt.ylabel('velocity (m/s)')
plt.legend()
plt.title('Velocity')

plt.subplot(1,3,3)
for i in range(1, N):
    err = x_hist[:, i] - (x_hist[:, 0] + r_i[i])
    plt.plot(t, err, label=f'vehicle {i}')
plt.xlabel('time (s)')
plt.ylabel('error (m)')
plt.axhline(0, color='k', linestyle='--')
plt.legend()
plt.title('Position error (relative to leader)')

plt.tight_layout()
plt.show()