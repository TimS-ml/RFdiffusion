# 推理流程详解

本文档详细说明 RFdiffusion 在推理时的完整执行流程。

## 🎯 总体流程

```
用户命令 → 参数解析 → 模型加载 → 初始化 → 采样循环 → 后处理 → 输出
```

---

## 📝 第一步：命令行调用

### 典型命令

```bash
python scripts/run_inference.py \
    inference.output_prefix=output/design \
    inference.input_pdb=target.pdb \
    'contigmap.contigs=[A1-100/0 50-100]' \
    inference.num_designs=10 \
    inference.ckpt_override_path=models/Base_ckpt.pt
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `output_prefix` | 输出文件前缀 | `output/my_design` |
| `input_pdb` | 输入PDB文件（可选） | `1abc.pdb` |
| `contigs` | 序列映射定义 | `[A1-50/0 100-150]` |
| `num_designs` | 生成数量 | `10` |
| `ckpt_override_path` | 模型权重文件 | `Base_ckpt.pt` |
| `potentials` | 引导势能 | `binder_ROG:1.0` |

### Contig 语法

```
格式: [chain_start-end/insertion_length ...]

示例:
'A1-100/0'        # A链1-100残基，固定（插入0）
'50-100'          # 设计50-100个残基
'B5-20/0 20'      # B链5-20固定，然后设计20个残基
'A1-10/10-20'     # A链1-10固定，设计10-20个残基
```

---

## 🏗️ 第二步：初始化

### 2.1 参数解析和配置

```python
# scripts/run_inference.py

def main():
    # 1. 加载配置文件
    config = OmegaConf.load('config/inference.yaml')

    # 2. 命令行覆盖
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli_config)

    # 3. 验证参数
    validate_config(config)
```

**关键配置项**:
```yaml
inference:
  num_designs: 10           # 生成多少个设计
  ckpt_override_path: ...   # 模型路径
  num_steps: 50             # 去噪步数
  temperature: 1.0          # 采样温度

diffuser:
  T: 200                    # 训练时的扩散步数
  schedule: linear          # 噪声调度

model:
  n_layers: 24              # 模型层数
  n_head: 16                # 注意力头数
```

### 2.2 模型加载

```python
# inference/model_runners.py

class Sampler:
    def __init__(self, conf):
        # 1. 创建模型
        self.model = RoseTTAFoldModule(
            n_layers=conf.model.n_layers,
            # ... 其他参数
        )

        # 2. 加载权重
        checkpoint = torch.load(conf.inference.ckpt_override_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 3. 设置为评估模式
        self.model.eval()

        # 4. 移至GPU
        self.model = self.model.to('cuda')

        # 5. 创建扩散器
        self.diffuser = Diffuser(T=200)
```

### 2.3 输入处理

```python
def load_input_data(conf):
    """处理输入PDB和contig定义"""

    # 1. 解析PDB（如果提供）
    if conf.inference.input_pdb:
        parsed = parse_pdb(
            conf.inference.input_pdb,
            parse_hetatom=False
        )
        xyz_fixed = parsed['xyz']      # 固定区域坐标
        seq_fixed = parsed['seq']      # 固定区域序列
    else:
        xyz_fixed = None
        seq_fixed = None

    # 2. 解析contig映射
    contig_map = ContigMap(
        parsed_pdb=parsed,
        contigs=conf.contigmap.contigs
    )

    # 3. 确定设计长度
    L_total = contig_map.contig_length
    L_fixed = len(contig_map.receptor)
    L_design = L_total - L_fixed

    return {
        'xyz_fixed': xyz_fixed,
        'seq_fixed': seq_fixed,
        'contig_map': contig_map,
        'L_total': L_total
    }
```

---

## 🔄 第三步：采样循环

### 3.1 采样初始化

```python
def sample_init(self, L):
    """
    初始化采样状态

    Args:
        L: 蛋白质长度

    Returns:
        初始状态字典
    """

    # 1. 从高斯噪声初始化坐标
    xyz_0 = torch.randn(1, L, 3) * 10.0  # 标准差10Å

    # 2. 从均匀分布初始化旋转
    R_0 = random_rotation(L)

    # 3. 组合为刚体变换（frames）
    frames_0 = {
        'xyz': xyz_0,      # CA位置
        'R': R_0,          # 局部坐标系方向
    }

    # 4. 初始化序列（全部mask）
    seq_0 = torch.full((1, L), 20)  # 20 = MASK_TOKEN

    # 5. 创建mask（哪些位置需要设计）
    mask_design = torch.ones(1, L, dtype=torch.bool)
    if self.contig_map:
        # 固定区域不设计
        mask_design[:, self.contig_map.receptor] = False

    return {
        'frames': frames_0,
        'seq': seq_0,
        'mask': mask_design,
        't': self.diffuser.T  # 从最大噪声开始
    }
```

### 3.2 主采样循环

```python
def sample(self, L, num_samples=1):
    """
    主采样函数

    Args:
        L: 蛋白质长度
        num_samples: 生成样本数
    """

    designs = []

    for n in range(num_samples):
        print(f"Generating design {n+1}/{num_samples}")

        # 1. 初始化
        state = self.sample_init(L)

        # 2. 采样循环（从T到1）
        for t in reversed(range(1, self.diffuser.T + 1)):
            state['t'] = t

            # 2.1 单步去噪
            state = self.sample_step(state)

            # 2.2 可选：保存轨迹
            if self.save_trajectory and t % 10 == 0:
                self.trajectory.append(state.copy())

            # 2.3 进度显示
            if t % 20 == 0:
                print(f"  Step {self.diffuser.T - t}/{self.diffuser.T}")

        # 3. 最终结构
        final_design = self.finalize(state)
        designs.append(final_design)

    return designs
```

### 3.3 单步去噪（核心）

```python
def sample_step(self, state):
    """
    单步去噪

    流程:
    1. 模型前向传播（预测噪声）
    2. 计算去噪方向
    3. （可选）应用势能引导
    4. 更新状态
    """

    frames_t = state['frames']
    seq = state['seq']
    t = state['t']

    # ===== 步骤1: 预处理 =====
    # 构建模型输入特征
    msa_feat, pair_feat, xyz_feat = self._preprocess(
        frames_t, seq, t
    )

    # ===== 步骤2: 模型预测 =====
    with torch.no_grad():
        model_out = self.model(
            msa=msa_feat,
            pair=pair_feat,
            xyz=xyz_feat,
            t=t / self.diffuser.T  # 归一化时间
        )

    # 模型输出
    score_trans = model_out['trans_score']  # 平移评分
    score_rot = model_out['rot_score']      # 旋转评分
    seq_logits = model_out['seq_logits']    # 序列预测

    # ===== 步骤3: 应用势能引导（可选）=====
    if self.potential_manager:
        # 计算势能梯度
        frames_t.requires_grad_(True)
        energy = self.potential_manager.compute_all_potentials(
            frames_t, t
        )
        grad_energy = torch.autograd.grad(energy, frames_t)[0]

        # 引导评分
        guide_scale = self.potential_manager.get_guide_scale(t)
        score_trans = score_trans - guide_scale * grad_energy['trans']
        score_rot = score_rot - guide_scale * grad_energy['rot']

    # ===== 步骤4: 去噪更新 =====
    frames_t_minus_1 = self.diffuser.p_sample(
        x_t=frames_t,
        t=t,
        score={'trans': score_trans, 'rot': score_rot},
        temperature=self.temperature
    )

    # ===== 步骤5: 序列更新（可选）=====
    if self.update_seq:
        seq_probs = F.softmax(seq_logits, dim=-1)
        seq = torch.multinomial(seq_probs.view(-1, 20), 1)
        seq = seq.view(1, -1)

    # ===== 步骤6: 固定区域（支架设计）=====
    if self.contig_map:
        # 保持固定区域不变
        mask_fixed = ~state['mask']
        frames_t_minus_1['xyz'][mask_fixed] = self.xyz_fixed[mask_fixed]
        frames_t_minus_1['R'][mask_fixed] = self.R_fixed[mask_fixed]

    # 更新状态
    state['frames'] = frames_t_minus_1
    state['seq'] = seq
    state['t'] = t - 1

    return state
```

### 3.4 预处理详解

```python
def _preprocess(self, frames, seq, t):
    """
    构建模型输入特征
    """
    B, L = seq.shape

    # 1. 序列特征
    seq_1hot = F.one_hot(seq, num_classes=21)  # (B, L, 21)

    # 2. 位置编码
    pos_enc = positional_encoding(L, d_model=256)  # (L, 256)
    pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)

    # 3. 时间嵌入
    t_emb = timestep_embedding(t, dim=256)  # (256,)
    t_emb = t_emb.unsqueeze(0).unsqueeze(0).expand(B, L, -1)

    # 4. MSA特征（单序列情况）
    msa_feat = torch.cat([
        seq_1hot,
        pos_enc,
        t_emb
    ], dim=-1)  # (B, L, 21+256+256)
    msa_feat = msa_feat.unsqueeze(1)  # (B, 1, L, feat_dim)

    # 5. Pair特征
    # 5.1 相对位置
    rel_pos = torch.arange(L)[:, None] - torch.arange(L)[None, :]
    rel_pos_feat = rbf_encode(rel_pos)  # (L, L, 36)

    # 5.2 相对方向（从frames计算）
    pair_feat = self._get_pair_features(frames)  # (B, L, L, feat_dim)

    # 5.3 组合
    pair_feat = torch.cat([
        rel_pos_feat.unsqueeze(0).expand(B, -1, -1, -1),
        pair_feat
    ], dim=-1)

    # 6. 结构特征（xyz坐标）
    xyz_feat = frames['xyz']  # (B, L, 3)

    return msa_feat, pair_feat, xyz_feat
```

### 3.5 后验采样（Diffuser）

```python
# diffusion.py

class Diffuser:
    def p_sample(self, x_t, t, score, temperature=1.0):
        """
        反向采样: x_t → x_{t-1}

        基于DDPM公式:
        x_{t-1} = μ(x_t, t) + σ(t) * z

        其中:
        - μ: 后验均值（从score计算）
        - σ: 后验标准差
        - z: 标准高斯噪声
        """

        # 1. 提取噪声调度参数
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_t_minus_1 = self.alpha_bar[t-1] if t > 1 else 1.0
        beta_t = self.beta[t]

        # 2. 计算后验均值
        # μ = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * score)
        coef1 = 1.0 / torch.sqrt(1.0 - beta_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)

        mu = coef1 * (x_t - coef2 * score)

        # 3. 计算后验标准差
        sigma = torch.sqrt(
            (1.0 - alpha_bar_t_minus_1) / (1.0 - alpha_bar_t) * beta_t
        )

        # 4. 采样（t=1时不加噪声）
        if t > 1:
            noise = torch.randn_like(x_t)
            x_t_minus_1 = mu + temperature * sigma * noise
        else:
            x_t_minus_1 = mu

        return x_t_minus_1

    def p_sample_rotation(self, R_t, t, score_rot, temperature=1.0):
        """
        旋转的反向采样（在SO(3)上）

        使用IGSO3分布
        """
        # 1. 从评分计算去噪方向（切空间）
        omega = self.igso3.score_to_omega(score_rot, t)

        # 2. 在SO(3)上更新
        # R_{t-1} = R_t * Exp(σ(t) * omega + noise)
        sigma = self.igso3.sigma(t)

        if t > 1:
            noise_omega = temperature * torch.randn_like(omega)
            delta_R = so3_exp(sigma * omega + noise_omega)
        else:
            delta_R = so3_exp(sigma * omega)

        R_t_minus_1 = R_t @ delta_R

        return R_t_minus_1
```

---

## 🎨 第四步：特殊设计模式

### 4.1 支架引导设计

```python
class ScaffoldedSampler(Sampler):
    """
    固定部分结构，设计其余部分

    应用场景:
    - 结合位点设计：固定target，设计binder
    - 片段嵌入：固定motif，设计周围支架
    """

    def sample_step(self, state):
        # 1. 正常预测
        state = super().sample_step(state)

        # 2. 恢复固定区域
        mask_fixed = self.contig_map.receptor

        # 平移
        state['frames']['xyz'][:, mask_fixed] = self.xyz_scaffold[mask_fixed]

        # 旋转
        state['frames']['R'][:, mask_fixed] = self.R_scaffold[mask_fixed]

        # 序列
        state['seq'][:, mask_fixed] = self.seq_scaffold[mask_fixed]

        return state
```

### 4.2 部分扩散

```python
class PartialDiffusionSampler(Sampler):
    """
    不从T开始，而从中间某个t_start开始

    用途:
    - 结构优化：t_start较小
    - 多样性生成：t_start较大
    """

    def sample_init(self, L):
        # 1. 加载初始结构
        xyz_init = self.initial_structure['xyz']

        # 2. 添加部分噪声（到t_start）
        t_start = self.partial_T
        frames_t = self.diffuser.q_sample(
            xyz_init,
            t=t_start
        )

        return {
            'frames': frames_t,
            't': t_start  # 从这里开始
        }
```

### 4.3 自条件化

```python
class SelfConditioning(Sampler):
    """
    使用前一次预测作为条件

    提高质量，但速度减半（每步2次前向）
    """

    def sample_step(self, state):
        frames_t = state['frames']
        t = state['t']

        # 第一次预测（无条件）
        with torch.no_grad():
            out_1 = self.model(
                frames=frames_t,
                t=t,
                self_cond=None
            )

        # 第二次预测（以第一次预测为条件）
        with torch.no_grad():
            out_2 = self.model(
                frames=frames_t,
                t=t,
                self_cond=out_1['frames_pred']
            )

        # 使用第二次预测去噪
        state['frames'] = self.diffuser.p_sample(
            frames_t, t, out_2['score']
        )

        return state
```

### 4.4 对称设计

```python
class SymmetricSampler(Sampler):
    """
    设计对称的蛋白质复合物
    """

    def __init__(self, conf, symdef='C3'):
        super().__init__(conf)
        self.symgen = SymGen(symdef)

        # 只设计asymmetric unit
        self.L_asu = conf.L_total // self.symgen.n_units

    def sample_init(self, L):
        # 初始化asymmetric unit
        state = super().sample_init(self.L_asu)

        # 应用对称生成完整复合物
        state['frames_full'] = self.symgen.apply(state['frames'])

        return state

    def sample_step(self, state):
        # 1. 只对asymmetric unit去噪
        frames_asu = state['frames'][:, :self.L_asu]

        # ... 预测和去噪 ...

        # 2. 应用对称
        frames_full = self.symgen.apply(frames_asu)

        # 3. 更新状态
        state['frames'] = frames_asu
        state['frames_full'] = frames_full

        return state
```

---

## 🔍 第五步：势能引导

### 5.1 引导原理

在采样过程中，通过势能函数的梯度引导生成：

```
score_guided = score_model - λ * ∇U(x)

其中:
- score_model: 模型预测的评分
- U(x): 势能函数
- λ: 引导强度
```

### 5.2 势能计算

```python
# 在sample_step中

# 1. 允许梯度计算
frames_t = state['frames']
frames_t.requires_grad_(True)

# 2. 计算所有势能
total_energy = 0
for potential in self.potentials:
    if potential.is_active(t):
        energy = potential(frames_t)
        total_energy += potential.weight * energy

# 3. 计算梯度
grad_energy = torch.autograd.grad(
    total_energy,
    frames_t
)[0]

# 4. 应用到评分
guide_scale = get_guide_scale(t)
score_guided = score_model - guide_scale * grad_energy
```

### 5.3 常用势能组合

#### 结合位点设计
```python
potentials = [
    "type:binder_ROG,weight:1.0,min_t:1,max_t:50",
    "type:binder_ncontacts,weight:3.0,min_t:1,max_t:40",
    "type:interface_ncontacts,weight:2.0,min_t:1,max_t:30",
]
```

#### 对称寡聚体设计
```python
potentials = [
    "type:monomer_ROG,weight:1.0",
    "type:olig_contacts,weight:2.0,min_t:10,max_t:60",
]
```

---

## 📤 第六步：后处理和输出

### 6.1 最终化

```python
def finalize(self, state):
    """
    完成采样后的处理
    """
    frames_final = state['frames']
    seq_final = state['seq']

    # 1. 构建全原子结构
    xyz_allatom = build_full_structure(
        frames=frames_final,
        seq=seq_final
    )

    # 2. 侧链优化（可选）
    if self.optimize_sidechain:
        xyz_allatom = optimize_sidechains(
            xyz_allatom, seq_final
        )

    # 3. 评分
    scores = {}
    scores['rmsd'] = calc_rmsd(xyz_allatom, self.reference)
    scores['clash'] = calc_clashes(xyz_allatom)

    return {
        'xyz': xyz_allatom,
        'seq': seq_final,
        'scores': scores
    }
```

### 6.2 写入PDB

```python
def save_design(design, output_path):
    """保存设计到PDB文件"""

    # 1. 写主PDB文件
    writepdb(
        filename=f"{output_path}.pdb",
        atoms=design['xyz'],
        seq=design['seq']
    )

    # 2. 写轨迹（如果有）
    if 'trajectory' in design:
        writepdb_multi(
            filename=f"{output_path}_traj.pdb",
            atoms_stack=design['trajectory'],
            seq_stack=design['seq']
        )

    # 3. 写元数据
    metadata = {
        'sequence': seq_to_string(design['seq']),
        'length': len(design['seq']),
        'scores': design['scores'],
        'timestamp': datetime.now().isoformat()
    }

    with open(f"{output_path}_meta.json", 'w') as f:
        json.dump(metadata, f, indent=2)
```

---

## 📊 完整执行时间线

### 示例：生成100残基蛋白（50步采样）

```
时间   操作                          GPU内存    说明
──────────────────────────────────────────────────────
0.0s   加载配置和模型                  2GB      一次性
0.5s   ├─ 创建模型实例
1.0s   ├─ 加载权重
1.2s   └─ 移至GPU

1.2s   初始化采样                      +1GB
1.3s   ├─ 采样初始噪声
1.4s   ├─ 构建输入特征
1.5s   └─ 初始化状态

1.5s   采样循环开始                    +5GB     主要计算
3.0s   ├─ Step 50/50 (t=50)           峰值8GB
4.5s   ├─ Step 40/50 (t=40)
6.0s   ├─ Step 30/50 (t=30)                   每步~30ms
7.5s   ├─ Step 20/50 (t=20)
9.0s   ├─ Step 10/50 (t=10)
10.5s  └─ Step 1/50  (t=1)

10.5s  后处理                          -4GB
11.0s  ├─ 构建全原子
11.3s  ├─ 侧链优化（可选）
11.5s  └─ 评分

11.5s  写入输出                        清理
11.7s  ├─ 写PDB
11.8s  ├─ 写轨迹
11.9s  └─ 写元数据

12.0s  完成                            释放GPU
```

### 性能影响因素

| 因素 | 影响 | 优化方法 |
|------|------|----------|
| 蛋白质长度 | O(L²) | 减少长度，分段设计 |
| 采样步数 | 线性 | 减少步数（50→25） |
| 批量大小 | GPU内存 | 使用梯度检查点 |
| 势能引导 | +20-50% | 限制活跃时间窗口 |
| 自条件化 | 2倍时间 | 仅关键步骤使用 |

---

## 🐛 调试和监控

### 关键检查点

```python
# 在sample_step中添加断言和日志

def sample_step(self, state):
    t = state['t']

    # 检查1: 坐标范围
    xyz = state['frames']['xyz']
    assert xyz.abs().max() < 1000, f"坐标爆炸 at t={t}"

    # 检查2: NaN检测
    assert not torch.isnan(xyz).any(), f"NaN坐标 at t={t}"

    # 检查3: 能量监控
    if t % 10 == 0:
        energy = self.potential_manager.compute_all_potentials(
            state['frames'], t
        )
        print(f"t={t}, energy={energy.item():.2f}")

    # ... 正常采样 ...

    return state
```

### 常见问题和解决

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| 坐标爆炸 | 梯度过大 | 减小学习率/温度 |
| NaN错误 | 数值不稳定 | 增加eps，检查除零 |
| GPU OOM | 内存不足 | 减小批量，梯度检查点 |
| 结构崩溃 | 引导过强 | 减小势能权重 |
| 收敛慢 | 步数不足 | 增加采样步数 |

---

## 🎯 优化技巧

### 1. 快速原型
```python
# 用于快速测试
config = {
    'num_steps': 10,      # 少量步数
    'num_designs': 1,     # 单个样本
    'L': 50,              # 短序列
}
```

### 2. 生产模式
```python
# 用于实际设计
config = {
    'num_steps': 50,      # 标准步数
    'num_designs': 100,   # 批量生成
    'temperature': 1.0,   # 标准温度
}
```

### 3. 高质量模式
```python
# 用于关键设计
config = {
    'num_steps': 200,         # 更多步数
    'self_conditioning': True, # 自条件化
    'optimize_sidechain': True,# 侧链优化
}
```

---

## 📚 下一步

- 了解 [模型架构](./04_model_architecture.md) 深入理解神经网络
- 阅读 [数学原理](./05_mathematical_foundations.md) 理解扩散模型理论
- 查看 [核心模块](./02_core_modules.md) 了解各组件详情
