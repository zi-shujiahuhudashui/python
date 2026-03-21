import json
import numpy as np

class CoordinateSystem:
    def __init__(self, basis_vectors, vector_coords):
        """
        basis_vectors: #这个表示基向量
        vector_coords: #这个表示向量坐标
        """
        # '.T' : .T 是 numpy 数组的转置属性，将矩阵的行和列互换。
        # 将基向量列表转换为矩阵，每一列是一个基向量
        # 这里引用了numpy库，用于转置
        self.basis = np.array(basis_vectors, dtype=float).T
        # 将向量坐标转为列向量
        '''
        vector_coords 是一个列表，比如 [1, 2]。
        np.array(vector_coords) 会生成一个一维数组（形状 (2,)）。
        为了把它当作列向量（形状 (2,1)），我们使用 reshape(-1, 1)。
        -1 表示自动计算该维度大小，这里变成 2 行 1 列。这样我们就可以进行矩阵乘法
        '''
        self.vector = np.array(vector_coords, dtype=float).reshape(-1, 1)
        self.dim = self.basis.shape[0]

        # 检查基矩阵是否为方阵且维度匹配
        if self.basis.shape != (self.dim, self.dim):
            raise ValueError("基向量必须构成方阵")
        if self.vector.shape[0] != self.dim:
            raise ValueError("向量维度必须与基矩阵维度匹配")
            # 即self.vector.shape[0]要等于self.basis.shape[0]，保持维度一致

    def is_valid(self):
        """检查基向量是否线性无关（行列式不为零）"""
        return abs(np.linalg.det(self.basis)) > 1e-10
    '''
    一个坐标系有效的条件是基向量线性无关。对于方阵，线性无关等价于行列式不为零。
    这里用了一个很小的阈值 1e-10 来避免浮点数精度问题。
    '''

    def transfer(self, target_basis_vectors):
        """
        将当前向量转移到目标坐标系。
        target_basis_vectors: list of lists, 目标坐标系的基向量（在绝对坐标系中的坐标）
        返回新坐标系中的坐标（列表形式），并更新当前坐标系和目标向量。
        """
        new_basis = np.array(target_basis_vectors, dtype=float).T
        if new_basis.shape != (self.dim, self.dim):
            raise ValueError("目标基维度不匹配")
        # 解线性方程组：new_basis * new_vector = self.vector_in_absolute?
        # 注意：self.vector 是当前坐标系下的坐标，它并不是绝对坐标。
        # 我们需要先得到绝对坐标，然后求新坐标系下的坐标。
        # 绝对坐标 = self.basis @ self.vector
        # 新坐标 = inv(new_basis) @ 绝对坐标
        # 所以新坐标 = inv(new_basis) @ self.basis @ self.vector
        # 但更简洁：新坐标 = solve(new_basis, self.basis @ self.vector)
        abs_vector = self.basis @ self.vector
        try:
            new_vector = np.linalg.solve(new_basis, abs_vector)
        except np.linalg.LinAlgError:
            raise ValueError("目标基矩阵奇异，无法转移")
        # 更新当前坐标系和目标向量
        self.basis = new_basis
        self.vector = new_vector
        return self.vector.flatten().tolist()

    def projection(self, axis_index):
        """计算当前向量在指定轴（基向量）上的投影长度"""
        # 先得到向量的绝对坐标
        abs_vector = self.basis @ self.vector
        # 指定轴的绝对坐标（即基向量）
        axis = self.basis[:, axis_index].reshape(-1, 1)
        # 投影长度 = (abs_vector · axis) / |axis|
        proj = np.dot(abs_vector.T, axis) / np.linalg.norm(axis)
        return float(proj)

    def angle(self, axis_index):
        """计算当前向量与指定轴的夹角（弧度）"""
        abs_vector = self.basis @ self.vector
        axis = self.basis[:, axis_index].reshape(-1, 1)
        dot = np.dot(abs_vector.T, axis)
        norm_v = np.linalg.norm(abs_vector)
        norm_axis = np.linalg.norm(axis)
        cos_theta = dot / (norm_v * norm_axis + 1e-12)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return float(np.arccos(cos_theta))

    def area_scaling(self):
        """计算当前坐标系的面积缩放因子（行列式绝对值）"""
        return float(abs(np.linalg.det(self.basis)))

    '''
       我去了要用到numpy这么多函数，这没ai辅助怎么可能一个人学完并记住到最后完成
    '''

def main():
    # 读取 JSON 文件
    with open('data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取数据
    vectors = data['vectors']            # 多个向量
    ori_axis = data['ori_axis']          # 初始基向量
    tasks = data['tasks']                # 任务列表

    # 对每个向量依次处理
    for idx, vec_coords in enumerate(vectors):
        print(f"\n========== 处理第 {idx+1} 个向量: {vec_coords} ==========")

        # 创建当前向量对应的坐标系对象（初始坐标系相同）
        cs = CoordinateSystem(ori_axis, vec_coords)

        if not cs.is_valid():
            print("初始坐标系无效，跳过此向量")
            continue

        # 执行任务列表
        for task in tasks:
            typ = task['type']
            if typ == 'change_axis':
                target_basis = task['obj_axis']
                try:
                    new_coords = cs.transfer(target_basis)
                    print(f"  转移后新坐标: {new_coords}")
                except Exception as e:
                    print(f"  转移失败: {e}")
            elif typ == 'axis_projection':
                # 计算在每个轴上的投影长度
                dim = cs.dim
                for i in range(dim):
                    proj = cs.projection(i)
                    print(f"  向量在第{i}轴上的投影长度: {proj:.6f}")
            elif typ == 'axis_angle':
                # 计算与每个轴的夹角
                dim = cs.dim
                for i in range(dim):
                    angle = cs.angle(i)
                    print(f"  向量与第{i}轴的夹角: {angle:.6f} 弧度")
            elif typ == 'area':
                area = cs.area_scaling()
                print(f"  当前坐标系的面积缩放因子: {area:.6f}")
            else:
                print(f"  未知任务类型: {typ}")

if __name__ == '__main__':
    main()