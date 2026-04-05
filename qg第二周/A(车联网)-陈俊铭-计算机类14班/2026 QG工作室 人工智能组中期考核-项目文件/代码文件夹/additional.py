import json
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
import matplotlib.animation as animation

# ==================== 参数设置
dt = 0.05
T = 30
steps = int(T / dt)
gamma = 1.0
spacing = 80.0
num_vehicles_per_road = 3

# 读取地图数据
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
points = data['points']
coords = np.array([[p['x'], p['y']] for p in points])

# 手动定义道路边（环形）
edges = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 0]
]

# 道路类
class Road:
    def __init__(self, start, end, num_vehicles, spacing):
        self.start = start
        self.end = end
        self.length = np.linalg.norm(end - start)
        self.dir = (end - start) / self.length
        self.x = np.linspace(0, self.length, num_vehicles)
        self.v = np.zeros(num_vehicles)
        self.spacing = spacing
        self.leader_v = 20.0

    def update(self, dt, gamma):
        u = np.zeros_like(self.v)
        pos_err_leader = self.x[0] - 0.0
        vel_err_leader = self.v[0] - self.leader_v
        u[0] = - (pos_err_leader + gamma * vel_err_leader)
        for i in range(1, len(self.x)):
            pos_err = (self.x[i] - self.x[i-1] - self.spacing)
            vel_err = self.v[i] - self.v[i-1]
            u[i] = - (pos_err + gamma * vel_err)
        self.x += self.v * dt
        self.v += u * dt
        self.x = np.clip(self.x, 0, self.length)

    def get_positions(self):
        return self.start + np.outer(self.x, self.dir)

# 初始化道路
roads = []
for (i, j) in edges:
    start = coords[i]
    end = coords[j]
    road = Road(start, end, num_vehicles_per_road, spacing)
    roads.append(road)

# 动画
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_aspect('equal')
ax.grid(True)

for (i, j) in edges:
    ax.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], 'k-', lw=1)

scats = []
for road in roads:
    scat = ax.scatter([], [], s=40, c='red', alpha=0.8)
    scats.append(scat)

x_min, x_max = coords[:,0].min()-300, coords[:,0].max()+300
y_min, y_max = coords[:,1].min()-300, coords[:,1].max()+300
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

for i, p in enumerate(points):
    ax.text(coords[i,0], coords[i,1], p['name'], fontsize=9, ha='center')

def animate(frame):
    for road in roads:
        road.update(dt, gamma)
    for scat, road in zip(scats, roads):
        pos = road.get_positions()
        scat.set_offsets(pos)
    return scats

ani = animation.FuncAnimation(fig, animate, frames=steps, interval=dt*1000, blit=False)
ani.save('map_formation.gif', writer='pillow', fps=20)
plt.show()