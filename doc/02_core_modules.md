# 核心模块详解

本文档详细介绍 RFdiffusion 中各个核心模块的功能和作用。

## 📁 模块组织

```
rfdiffusion/
├── 基础模块（化学和几何）
│   ├── chemical.py
│   ├── util.py
│   ├── coords6d.py
│   └── kinematics.py
├── 扩散相关
│   ├── diffusion.py
│   └── igso3.py
├── 模型架构
│   ├── RoseTTAFoldModel.py
│   ├── SE3_network.py
│   ├── Attention_module.py
│   ├── Track_module.py
│   └── Embeddings.py
├── 推理引擎
│   └── inference/
│       ├── model_runners.py
│       ├── utils.py
│       └── symmetry.py
├── 引导机制
│   └── potentials/
│       ├── manager.py
│       └── potentials.py
└── 辅助模块
    ├── contigs.py
    ├── scoring.py
    └── AuxiliaryPredictor.py
```

---

## 🔬 基础模块

### chemical.py - 化学常量库

**作用**: 定义所有氨基酸的化学性质和原子信息

**核心数据结构**:

```python
# 22种氨基酸（20标准 + UNK + MAS）
num2aa = ['ALA', 'ARG', ..., 'UNK', 'MAS']

# 原子命名（全原子表示，27个原子位置）
aa2long = [
    # ALA: N, CA, C, O, CB, ..., 氢原子
    (" N  ", " CA ", " C  ", " O  ", " CB ", ...),
    ...
]

# 键连接关系
aabonds = [
    ((" N  ", " CA "), (" CA ", " C  "), ...),  # ALA的键
    ...
]

# 理想坐标（Rosetta参数）
ideal_coords = [...]
```

**关键功能**:
- 氨基酸索引和名称转换
- 原子类型和性质定义
- 理想几何参数
- 用于评分的 LJ/LK 参数

**使用场景**: 所有需要处理蛋白质化学信息的地方

---

### util.py - 结构操作工具

**作用**: 提供蛋白质结构的核心操作函数

**关键函数**:

#### 1. 几何计算
```python
generate_Cbeta(N, Ca, C)
# 从主链原子重建 CB 位置
# 用途: GLY 的 CB 重建，验证侧链

rigid_from_3_points(N, Ca, C)
# 从 N-CA-C 构建局部坐标系
# 返回: 旋转矩阵 R 和平移 T
```

#### 2. 扭转角计算
```python
get_torsions(xyz, seq, ...)
# 计算所有扭转角（omega, phi, psi, chi1-4）
# 返回: (cos, sin) 表示，避免角度不连续

get_tor_mask(seq, torsion_indices)
# 生成有效扭转角的掩码
# 考虑氨基酸类型和缺失原子
```

#### 3. PDB 输入/输出
```python
writepdb(filename, atoms, seq, ...)
# 写入 PDB 文件
# 支持多种原子表示: CA, 主链, 全原子

writepdb_multi(filename, atoms_stack, ...)
# 写入多个构象到一个 PDB
# 用于轨迹可视化
```

#### 4. 预计算表
```python
# 在模块加载时预计算
tip_indices        # 每个氨基酸的尖端原子
torsion_indices    # 扭转角定义
num_bonds          # 原子间键数（距离矩阵）
ljlk_parameters    # LJ 和 LK 参数
hbtypes, hbpolys   # 氢键参数
```

**使用场景**:
- 推理时的结构更新
- PDB 文件读写
- 结构验证和分析

---

### kinematics.py - 蛋白质运动学

**作用**: 从扭转角构建全原子坐标

**核心算法**:
```
扭转角 (phi, psi, chi) → 主链框架 → 侧链原子位置
```

**关键函数**:

```python
xyz_to_c6d(xyz, mask)
# 3D坐标 → 6D表示（距离+角度）
# 用于模型输入的特征化

c6d_to_bins(c6d, ...)
# 6D → 离散化的bins
# 用于预测目标

xyz_to_t2d(xyz, mask)
# 计算扭转角的2D表示
```

**6D 坐标表示**:

对每一对残基 (i, j):
```
距离: dist
方向角:
  - omega: CA(i) → CA(j) 相对于 N(i)-CA(i)-C(i) 平面
  - theta: CA(i) → CA(j) 相对于 CA(i)-CB(i)
  - phi: 两个局部坐标系的扭转角
```

**使用场景**: 特征提取、坐标重建

---

## 🌊 扩散相关

### diffusion.py - 扩散过程实现

**作用**: 实现前向加噪和反向去噪的核心逻辑

**核心类**:

#### 1. Diffuser - 主扩散类
```python
class Diffuser:
    def __init__(self, T=200, b_0=1e-4, b_T=0.02):
        # T: 扩散步数
        # b_0, b_T: 噪声调度参数
        self.T = T
        self.alpha_bar = compute_alpha_bar(...)  # 累积噪声系数
```

**关键方法**:
```python
# 前向扩散（训练时）
q_sample(x_0, t)
# x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise

# 反向采样（推理时）
p_sample(x_t, t, model_output)
# x_{t-1} = mu(x_t, model_output) + sigma * z
```

#### 2. IGSO3 - SO(3) 旋转扩散
```python
class IGSO3:
    # Isotropic Gaussian on SO(3)
    # 对旋转矩阵的扩散

    sample(t, N)
    # 采样噪声旋转

    score(R_t, R_0, t)
    # 计算评分函数（去噪方向）
```

**扩散调度**:
```python
# Beta 调度（控制噪声添加速度）
- linear: beta_t = b_0 + (b_T - b_0) * t/T
- cosine: 更平滑的调度
- sqrt: 开方调度
```

**组合扩散**:
```python
# 同时对平移和旋转扩散
diffuse_pose(frames, t):
    # frames: (R, t) 刚体变换
    trans = diffuse_translation(frames.t, t)
    rot = diffuse_rotation(frames.R, t)
    return combine(trans, rot)
```

**使用场景**:
- 训练时: 生成不同噪声水平的样本
- 推理时: 逐步去噪

---

### igso3.py - SO(3) 数学库

**作用**: 实现 SO(3) 李群上的操作

**核心函数**:

```python
# Lie 代数操作
hat(v)          # R^3 → so(3) (skew-symmetric)
vee(Omega)      # so(3) → R^3

# 指数/对数映射
Exp(omega)      # so(3) → SO(3)
Log(R)          # SO(3) → so(3)

# IGSO3 分布
f_igso3(omega, t)          # PDF
score_igso3(R, R0, t)      # Score function
sample_igso3(t, n)         # 采样
```

**数学背景**:

SO(3) 是 3D 旋转群，扩散在其上进行:
- 使用**测地距离**（geodesic distance）
- 保持旋转的流形结构
- 通过 Lie 代数实现高效计算

**使用场景**: `diffusion.py` 中的旋转扩散

---

## 🏗️ 模型架构

### RoseTTAFoldModel.py - 主模型

**作用**: 实现完整的 RoseTTAFold 架构

**模型结构**:

```python
class RoseTTAFoldModule(nn.Module):
    def __init__(self):
        # 嵌入层
        self.msa_emb = MSA_emb(...)
        self.pair_emb = ...
        self.templ_emb = ...

        # 主干网络（多层IterBlock）
        self.blocks = nn.ModuleList([
            IterBlock(...) for _ in range(n_layers)
        ])

        # 结构预测头
        self.str_refiner = SE3TransformerWrapper(...)
```

**前向传播**:
```python
def forward(self, msa, seq, xyz_t, t):
    # 1. 嵌入
    msa_feat = self.msa_emb(msa)
    pair_feat = self.pair_emb(seq)

    # 2. 迭代更新（三轨并行）
    for block in self.blocks:
        msa_feat, pair_feat, xyz = block(
            msa_feat, pair_feat, xyz_t
        )

    # 3. 结构细化
    xyz_pred = self.str_refiner(xyz, pair_feat)

    return xyz_pred
```

**关键特性**:
- **循环连接 (Recycling)**: 前一次预测作为输入
- **梯度检查点**: 节省内存
- **时间条件**: 通过 t 调节去噪强度

---

### SE3_network.py - SE(3)等变网络

**作用**: 处理 3D 结构，保持旋转/平移等变性

**网络结构**:
```python
class SE3TransformerWrapper(nn.Module):
    # SE(3)-Transformer:
    # 输入: 节点特征 + 边特征 + 3D坐标
    # 输出: 更新的特征 + 坐标
```

**等变性**:

旋转输入 → 旋转输出（保持几何关系）
```
R · f(x) = f(R · x)  for all R ∈ SO(3)
```

**实现方式**:
- 使用**球谐函数**表示方向
- 保持不同**度数** (degrees) 的特征分离
- Clebsch-Gordan 耦合实现等变操作

**使用场景**:
- 结构轨的更新
- 预测坐标偏移

---

### Attention_module.py - 注意力机制

**作用**: 实现各种注意力模块

**核心模块**:

#### 1. 标准注意力
```python
class Attention(nn.Module):
    # Q, K, V attention
    attn = softmax(QK^T / sqrt(d)) V
```

#### 2. 带偏置注意力
```python
class AttentionWithBias(nn.Module):
    # 添加 pair 信息作为偏置
    attn = softmax(QK^T / sqrt(d) + bias) V
```

#### 3. MSA 专用注意力
```python
class MSARowAttentionWithBias:
    # 沿序列位置的注意力

class MSAColAttention:
    # 沿MSA深度的注意力（进化信息）
```

**使用场景**: Track 模块中的信息传递

---

### Track_module.py - 三轨架构

**作用**: 实现 MSA/Pair/Structure 三轨的信息交换

**核心类**:

```python
class IterBlock(nn.Module):
    """单个迭代块，包含三轨更新"""

    def forward(self, msa, pair, xyz):
        # 1. MSA → Pair
        pair = pair + MSA2Pair(msa)

        # 2. Pair → MSA
        msa = msa + Pair2MSA(pair)

        # 3. Pair + Structure → Structure
        xyz = xyz + Str2Str(xyz, pair)

        # 4. Structure → Pair
        pair = pair + PairStr2Pair(xyz, pair)

        return msa, pair, xyz
```

**信息流**:
```
MSA ←→ Pair ←→ Structure
 ↓      ↓         ↓
自注意力 自注意力  SE(3)-Transformer
```

**关键特性**:
- **双向信息流**: 各轨道相互增强
- **残差连接**: 保持信息流通
- **LayerNorm**: 稳定训练

---

### Embeddings.py - 特征嵌入

**作用**: 将输入转换为神经网络可处理的特征

**核心模块**:

```python
class MSA_emb(nn.Module):
    # MSA → embedding
    # 氨基酸类型 → 向量

class Extra_emb(nn.Module):
    # 额外特征（距离、角度等）

class Templ_emb(nn.Module):
    # 模板结构特征

class Recycling(nn.Module):
    # 前一次预测的特征
```

---

## 🎯 推理引擎

### inference/model_runners.py - 推理运行器

**作用**: 管理完整的采样流程

**核心类**:

#### 1. Sampler - 基础采样器
```python
class Sampler:
    def __init__(self, conf):
        self.model = load_model(conf)
        self.diffuser = Diffuser(T=200)

    def sample(self):
        # 初始化
        xyz_t = sample_noise()

        # 采样循环
        for t in reversed(range(T)):
            # 模型预测
            pred = self.model(xyz_t, t)

            # 去噪
            xyz_t = self.diffuser.p_sample(
                xyz_t, t, pred
            )

        return xyz_t  # 最终结构
```

#### 2. ScaffoldedSampler - 支架引导
```python
class ScaffoldedSampler(Sampler):
    # 固定部分结构，设计其余部分

    def sample_step(self, xyz_t, t):
        # 预测
        pred = self.model(xyz_t, t)

        # 只更新可设计区域
        xyz_t[designable] = denoise(
            xyz_t[designable], pred
        )

        # 保持固定区域不变
        xyz_t[fixed] = xyz_scaffold[fixed]

        return xyz_t
```

#### 3. SelfConditioning - 自条件化
```python
class SelfConditioning(Sampler):
    # 用前一步的预测作为条件
    # 提高采样质量

    def sample_step(self, xyz_t, t):
        # 两次前向传播
        pred_0 = self.model(xyz_t, t, cond=None)
        pred_1 = self.model(xyz_t, t, cond=pred_0)

        return denoise(xyz_t, pred_1)
```

**采样参数**:
```python
# 常用参数
num_steps: 推理步数（50-200）
temperature: 采样温度（控制随机性）
self_cond: 是否使用自条件化
partial_T: 部分扩散的起始步数
```

---

### inference/utils.py - 推理工具

**作用**: 提供推理所需的各种辅助函数

**核心功能**:

#### 1. 去噪类
```python
class Denoise:
    """处理反向扩散的具体计算"""

    def get_next_frames(self, frames_t, t):
        # 计算 x_{t-1} 从 x_t

        # 1. 模型预测
        score = self.model(frames_t, t)

        # 2. 计算后验均值
        mu = self.get_mu_xt_x0(
            frames_t, score, t
        )

        # 3. 添加噪声（非最后一步）
        if t > 0:
            sigma = self.get_sigma(t)
            noise = sample_noise()
            frames_t_minus_1 = mu + sigma * noise
        else:
            frames_t_minus_1 = mu

        return frames_t_minus_1
```

#### 2. PDB 解析
```python
def parse_pdb(pdb_file, **kwargs):
    # 解析 PDB 文件
    # 返回: xyz坐标, 序列, mask等

def process_target(target_pdb, contigs):
    # 处理目标结构（用于binder设计）
    # 返回: 格式化的特征张量
```

#### 3. 噪声调度
```python
def get_noise_schedule(T, schedule_type):
    # 生成噪声调度
    # 支持: linear, cosine, sqrt

    if schedule_type == 'cosine':
        s = 0.008
        alpha_bar = cos((t/T + s)/(1+s) * pi/2)^2
    ...
    return alpha_bar
```

---

### inference/symmetry.py - 对称性处理

**作用**: 生成和维护对称的蛋白质复合物

**核心类**:

```python
class SymGen:
    """对称生成器"""

    def __init__(self, sym_type='C3'):
        # C3: 3重循环对称
        # D4: 4重二面体对称
        # O: 八面体对称
        # T: 四面体对称
        # I: 二十面体对称

        self.sym_type = sym_type
        self.rots = self.get_rotations()
```

**对称类型**:

#### 1. 循环对称 (Cn)
```python
def _apply_cyclic(self, xyz, n):
    # n重绕Z轴旋转
    rot_angle = 2*pi / n

    copies = []
    for i in range(n):
        R = rotation_z(i * rot_angle)
        copies.append(R @ xyz)

    return concat(copies)
```

#### 2. 二面体对称 (Dn)
```python
def _apply_dihedral(self, xyz, n):
    # 循环对称 + 镜面翻转
    cyclic = self._apply_cyclic(xyz, n)
    flipped = flip_y(cyclic)
    return concat([cyclic, flipped])
```

#### 3. 多面体对称
```python
# 预定义的旋转矩阵
octahedral_rotations = [...]  # 24个旋转
tetrahedral_rotations = [...]  # 12个旋转
icosahedral_rotations = [...]  # 60个旋转
```

**使用方法**:
```python
# 推理时应用对称
symgen = SymGen('C3')

for t in reversed(range(T)):
    # 1. 只预测/更新asymmetric unit
    xyz_asu = xyz_t[:len_asu]
    pred = model(xyz_asu, t)
    xyz_asu = denoise(xyz_asu, pred, t)

    # 2. 应用对称生成完整复合物
    xyz_t = symgen.apply(xyz_asu)
```

---

## ⚡ 引导机制

### potentials/manager.py - 势能管理器

**作用**: 协调多个势能函数的计算和应用

**核心类**:

```python
class PotentialManager:
    def __init__(self, potentials_config):
        # 解析配置字符串
        # 例如: "type:binder_ROG,weight:1.0,min_t:1,max_t:20"

        self.potentials = []
        for config in potentials_config:
            pot = self.create_potential(config)
            self.potentials.append(pot)

    def compute_all_potentials(self, xyz, t):
        """计算所有势能的总和"""
        total_energy = 0

        for pot in self.potentials:
            if pot.is_active(t):  # 检查时间窗口
                energy = pot(xyz)
                scale = self.get_guide_scale(t)
                total_energy += pot.weight * scale * energy

        return total_energy
```

**调度策略**:
```python
def get_guide_scale(self, t):
    # 随时间变化的缩放
    # 早期: 强引导
    # 后期: 弱引导（让模型主导）

    if t > 50:
        return 10.0  # 强引导
    elif t > 20:
        return 5.0   # 中等引导
    else:
        return 1.0   # 弱引导
```

---

### potentials/potentials.py - 具体势能

**作用**: 定义各种可微分的势能函数

**基类**:
```python
class Potential:
    """所有势能的基类"""

    def __call__(self, xyz):
        # 计算能量（可微分）
        raise NotImplementedError

    def is_active(self, t):
        # 是否在当前时间步活跃
        return self.min_t <= t <= self.max_t
```

**常用势能**:

#### 1. 回旋半径 (ROG)
```python
class monomer_ROG(Potential):
    """控制蛋白紧凑性"""

    def __call__(self, xyz):
        # 计算回旋半径
        center = xyz.mean(dim=0)
        rog = torch.sqrt(
            ((xyz - center)**2).sum()
        )

        # 惩罚偏离目标值
        return (rog - self.target_rog)**2
```

#### 2. 接触数
```python
class binder_ncontacts(Potential):
    """最大化binder和target的接触"""

    def __call__(self, xyz_binder, xyz_target):
        # 计算距离矩阵
        dist = cdist(xyz_binder, xyz_target)

        # 计数接触（距离 < 阈值）
        contacts = (dist < 8.0).float().sum()

        # 负号: 最大化接触
        return -contacts
```

#### 3. 对称接触
```python
class olig_contacts(Potential):
    """在对称界面维持接触"""

    def __call__(self, xyz_oligomer):
        # 对每对单体
        contacts_total = 0
        for i in range(n_monomers):
            for j in range(i+1, n_monomers):
                xyz_i = xyz_oligomer[i]
                xyz_j = xyz_oligomer[j]

                # 计算界面接触
                contacts = count_contacts(xyz_i, xyz_j)
                contacts_total += contacts

        return -contacts_total  # 最大化
```

**使用示例**:
```python
# 配置多个势能
potentials = [
    "type:binder_ROG,weight:2.0",
    "type:binder_ncontacts,weight:5.0,min_t:1,max_t:50",
]

manager = PotentialManager(potentials)

# 在采样循环中
for t in range(T, 0, -1):
    # 模型预测
    score = model(xyz_t, t)

    # 计算势能梯度
    energy = manager.compute_all_potentials(xyz_t, t)
    grad = torch.autograd.grad(energy, xyz_t)[0]

    # 引导去噪
    score_guided = score - grad
    xyz_t = denoise(xyz_t, score_guided, t)
```

---

## 🔧 辅助模块

### contigs.py - 序列映射

**作用**: 处理复杂的序列映射关系（支架、链断裂等）

**核心类**:
```python
class ContigMap:
    """
    映射关系管理:
    - 哪些残基来自模板（固定）
    - 哪些残基需要设计（可变）
    - 多链结构的处理
    """

    def __init__(self, contigs):
        # 解析 contig 字符串
        # 例如: "A1-10/0 B20-30/0 50-60"
        #       ^^固定^^ ^^固定^^ ^^设计^^
```

---

### scoring.py - 评分参数

**作用**: 提供基于物理的评分参数（来自Rosetta）

包含:
- Lennard-Jones 参数
- Lazaridis-Karplus 溶剂化能
- 氢键参数和多项式
- 原子类型定义

**使用场景**:
- 后处理阶段的结构评分
- 可选的能量引导

---

## 📊 模块依赖关系

```
                    run_inference.py
                          │
              ┌───────────┴───────────┐
              │                       │
    inference/model_runners.py   potentials/
              │                       │
              ├──────────┬────────────┤
              │          │            │
    RoseTTAFoldModel  diffusion.py  manager.py
              │          │            │
      ┌───────┼──────────┤            │
      │       │          │            │
   SE3_net Track_mod Embeddings   potentials.py
      │       │          │
      └───────┴──────────┘
              │
        ┌─────┴─────┐
        │           │
    util.py    chemical.py
```

## 🎯 使用建议

**快速理解代码**:
1. 先读 `diffusion.py` 理解扩散原理
2. 再读 `inference/model_runners.py` 理解采样流程
3. 然后读 `RoseTTAFoldModel.py` 理解模型结构

**修改和扩展**:
- 添加新势能: 在 `potentials/potentials.py`
- 修改采样策略: 在 `inference/model_runners.py`
- 调整模型架构: 在 `Track_module.py` 或 `SE3_network.py`
