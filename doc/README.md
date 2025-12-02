# RFdiffusion 文档

欢迎阅读 RFdiffusion 代码库文档！本文档旨在帮助不熟悉本项目的开发者快速理解代码结构和工作原理。

## 📚 文档目录

- **[项目概览](./01_overview.md)** - RFdiffusion 的基本介绍和设计理念
- **[核心模块](./02_core_modules.md)** - 关键文件和模块的详细说明
- **[推理流程](./03_inference_flow.md)** - 模型推理时的完整执行流程
- **[模型架构](./04_model_architecture.md)** - 神经网络架构详解
- **[数学原理](./05_mathematical_foundations.md)** - 扩散模型的数学基础

## 🚀 快速开始

如果你是第一次接触 RFdiffusion：

1. 从 **[项目概览](./01_overview.md)** 开始，了解整体架构
2. 阅读 **[推理流程](./03_inference_flow.md)**，理解模型如何生成蛋白质结构
3. 查看 **[核心模块](./02_core_modules.md)**，深入了解各个组件

## 📖 关于 RFdiffusion

RFdiffusion 是一个基于扩散模型的蛋白质结构设计工具。它通过学习蛋白质结构的分布，能够从头设计全新的蛋白质结构，或在给定模板的基础上进行结构优化。

### 主要功能

- **从头设计 (De novo design)**: 生成全新的蛋白质结构
- **支架设计 (Scaffolding)**: 在给定骨架上设计蛋白质
- **结合位点设计 (Binder design)**: 设计能与目标蛋白结合的结构
- **对称设计**: 支持各种对称性（循环、二面体、多面体）
- **引导设计**: 通过势能函数引导设计过程

## 🏗️ 代码结构概览

```
rfdiffusion/
├── __init__.py              # 包初始化
├── chemical.py              # 氨基酸和原子的化学常量
├── util.py                  # 结构操作工具函数
├── diffusion.py             # 扩散过程实现
├── RoseTTAFoldModel.py      # 主模型架构
├── SE3_network.py           # SE(3)等变网络
├── Attention_module.py      # 注意力机制
├── Track_module.py          # 三轨架构
├── Embeddings.py            # 特征嵌入
├── inference/               # 推理相关
│   ├── model_runners.py     # 推理运行器
│   ├── utils.py             # 推理工具
│   └── symmetry.py          # 对称性处理
└── potentials/              # 势能函数
    ├── manager.py           # 势能管理器
    └── potentials.py        # 具体势能函数
```

## 💡 核心概念

### 扩散模型

RFdiffusion 使用扩散模型来生成蛋白质结构：
- **前向过程**: 逐步向结构添加噪声
- **反向过程**: 从噪声中逐步恢复结构
- **SE(3)等变**: 保持旋转和平移不变性

### 三轨架构

模型使用三个相互作用的信息流：
- **MSA轨 (MSA track)**: 处理多序列比对信息
- **Pair轨 (Pair track)**: 处理残基对信息
- **结构轨 (Structure track)**: 处理3D几何信息

## 🔧 开发指南

### 添加新功能

1. 在 `potentials/` 下添加新的势能函数
2. 在 `inference/model_runners.py` 中添加新的采样策略
3. 扩展对称性支持在 `inference/symmetry.py`

### 调试技巧

- 使用 `model_input_logger.py` 记录模型输入
- 检查 `util.py` 中的 PDB 写入函数来验证结构
- 使用 `--write_trajectory` 查看采样过程

## 📝 引用

如果使用 RFdiffusion，请引用：
```
Watson et al. "De novo design of protein structure and function with RFdiffusion"
Nature (2023)
```

## 🤝 贡献

欢迎贡献代码和文档！请确保：
- 添加详细的 docstring
- 遵循现有的代码风格
- 更新相关文档
