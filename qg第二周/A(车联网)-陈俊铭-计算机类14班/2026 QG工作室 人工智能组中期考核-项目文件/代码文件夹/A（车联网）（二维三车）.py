import numpy as np
import matplotlib.pyplot as plt

# 仿真参数
N = 3                     # 车辆数（索引0是领导，1和2是跟随）
dt = 0.01                 # 仿真步长 (s)
T = 20                    # 总仿真时间 (s)
steps = int(T/dt)         # 总步数
beta = 1.0                # 邻居速度差增益
gamma = 1.0               # 领导速度差增益

# 通信拓扑
# 邻接矩阵 A: A[i][j]=1 表示车 i 能收到车 j 的信息
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

# 领导连接标志 k: k[i]=1 表示车 i 能收到领导（车0）的信息
k = np.array([0, 1, 1])

# 编队形状
# 二维扩展：目标位置现在是二维向量 (x, y)
# 定义每个车相对于领导的目标位置（二维）
# 例如：领导在 (0,0)，车1在后方2米 (0,-2)，车2在右后方 (2,-2)
target_positions = np.array([
    [0, 0],     # 领导车
    [0, -2],    # 跟随车1
    [2, -2]     # 跟随车2
])

# 二维扩展：r_ij 现在是 N×N×2 的数组，每个元素是二维向量
r_ij = np.zeros((N, N, 2))
for i in range(N):
    for j in range(N):
        r_ij[i, j] = target_positions[i] - target_positions[j]  # 满足反对称

# 与领导的期望间距 r_i (二维)
r_i = target_positions.copy()

# 初始状态
# 二维扩展：位置和速度现在是 N×2 的数组
x = np.array([
    [0.0, 0.0],    # 领导在原点
    [0.0, 5.0],    # 跟随车1 在领导正前方5米
    [3.0, 5.0]     # 跟随车2 在领导右前方5米
])
v = np.array([
    [1.0, 0.0],    # 领导车速度：x方向1，y方向0
    [0.0, 0.0],    # 跟随车1 初始静止
    [0.0, 0.0]     # 跟随车2 初始静止
])

# 领导车的控制输入（匀速，加速度为0）
u_leader = np.array([0.0, 0.0])

# 记录历史（每个时间步存储 N×2 的副本）
x_hist = [x.copy()]
v_hist = [v.copy()]

stable_count = 0

# 仿真循环
for step in range(steps):
    # 二维扩展：控制输入也是 N×2
    # 在二维代码中，状态量（位置、速度）变成了 N×2 的数组，每个车的运动分为 x 和 y 两个方向，
    # 因此控制输入也需要存储每个方向的加速度。所以 u 必须定义为 (N,2) 的二维数组
    u = np.zeros((N, 2))

    for i in range(N):
        # 邻居项
        neighbor_term = np.zeros(2)      # 二维扩展：邻居项是二维向量
        for j in range(N):
            if A[i, j] != 0:
                # 二维扩展：向量减法
                pos_err = x[i] - x[j] - r_ij[i, j]    # 位置误差（二维）
                vel_err = beta * (v[i] - v[j])         # 速度差（二维）
                neighbor_term += pos_err + vel_err
        u[i] -= neighbor_term

        # 领导项
        if k[i] != 0:
            # 二维扩展：向量运算
            pos_err_leader = x[i] - x[0] - r_i[i]
            vel_err_leader = gamma * (v[i] - v[0])
            u[i] -= k[i] * (pos_err_leader + vel_err_leader)

    # 领导车控制输入单独设定
    u[0] = u_leader  # 强制覆盖领导车的值

    # 欧拉更新（向量形式）
    x = x + v * dt
    v = v + u * dt

    x_hist.append(x.copy())
    # 自适应结束判断
    # 计算所有跟随车的最大误差（相对于期望位置）
    max_error = 0
    for i in range(1, N):  # 只检查跟随车
        err = np.linalg.norm(x[i] - (x[0] + r_i[i]))  # 二维欧氏距离
        max_error = max(max_error, err)

    # 如果误差小于阈值，累加稳定步数
    if max_error < 0.05:  # 阈值 5cm
        stable_count += 1
        if stable_count > 100:  # 连续 100 步稳定（即 1 秒）
            print(f"Formation achieved at t = {step * dt:.2f} s")
            v_hist.append(v.copy())
            break
    else:
        stable_count = 0
    v_hist.append(v.copy())

# 转换为数组以便绘图
x_hist = np.array(x_hist)   # shape: (steps+1, N, 2)
v_hist = np.array(v_hist)   # shape: (steps+1, N, 2)
actual_steps = len(x_hist)
t = np.linspace(0, (actual_steps-1)*dt, actual_steps)

# ---------------------------------------------------------------------------------------------------

# 绘图
# Fig 4: 二维轨迹图
plt.figure(figsize=(6, 6))
for i in range(N):
    plt.plot(x_hist[:, i, 0], x_hist[:, i, 1], label=f'vehicle {i}')
plt.scatter(x_hist[-1, :, 0], x_hist[-1, :, 1], c='red', s=50, marker='o')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trajectories (Fig 4)')
plt.axis('equal')
plt.legend()
plt.grid()
plt.savefig('Fig4_trajectories.png', dpi=150)
plt.show()

# Fig 5: 纵向和横向误差
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
for i in range(1, N):
    err_x = x_hist[:, i, 0] - (x_hist[:, 0, 0] + r_i[i, 0])
    err_y = x_hist[:, i, 1] - (x_hist[:, 0, 1] + r_i[i, 1])
    ax1.plot(t, err_x, label=f'vehicle {i}')
    ax2.plot(t, err_y, label=f'vehicle {i}')
ax1.axhline(0, color='k', linestyle='--')
ax2.axhline(0, color='k', linestyle='--')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('longitudinal error (m)')
ax1.legend()
ax2.set_xlabel('time (s)')
ax2.set_ylabel('lateral error (m)')
ax2.legend()
plt.suptitle('Position errors relative to leader (Fig 5)')
plt.tight_layout()
plt.savefig('Fig5_errors.png', dpi=150)
plt.show()

# Fig 6: 速度曲线（分 x 和 y）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
for i in range(N):
    ax1.plot(t, v_hist[:, i, 0], label=f'vehicle {i}')
    ax2.plot(t, v_hist[:, i, 1], label=f'vehicle {i}')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('x-velocity (m/s)')
ax1.legend()
ax2.set_xlabel('time (s)')
ax2.set_ylabel('y-velocity (m/s)')
ax2.legend()
plt.suptitle('Velocity evolution (Fig 6)')
plt.tight_layout()
plt.savefig('Fig6_velocities.png', dpi=150)
plt.show()
# 动图
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 10)
ax.set_aspect('equal')
ax.grid(True)

# 初始化画图元素
scats = [ax.plot([], [], 'o', markersize=10)[0] for _ in range(N)]
def animate(frame):
    for i in range(N):
        scats[i].set_data(x_hist[frame, i, 0], x_hist[frame, i, 1])
    ax.set_title(f't = {frame * dt:.2f} s')
    return scats

ani = animation.FuncAnimation(fig, animate, frames=range(0, len(x_hist), 10), interval=50, blit=False)
ani.save('formation.gif', writer='pillow', fps=20)
plt.close()