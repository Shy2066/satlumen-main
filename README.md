# SatLumen main experiment

本项目只包含 SatLumen 在干净 Fiber/Starlink 数据上的主试验：
输入处理、方向计数、训练、源域验证选模及四方向评估。
三种子为 42、2024、2025，共 6 次训练、12 项评估；不包含其他攻击方法。

## 环境

从本项目根目录执行：

```bash
conda env create -f environment.yml
conda activate satlumen-main
```

环境文件固定 Python 3.10.19、PyTorch 2.1.2+cu118 和本次检查使用的依赖版本。
其余依赖版本来自当前本地环境核对，不宣称是历史实验的完整环境快照。
GPU 实验需要支持 CUDA 11.8 的驱动；原论文使用 RTX 3090。
尚未在全新环境中执行依赖下载安装。

## 文件结构

```text
configs/          训练配置和正式评估配置
data_process/     轨迹转换、固定划分、计数缓存
src/              主模型、数据读取、训练器和工具
scripts/          主流程、单任务训练与评估
```

`configs/default.yaml` 用于训练编排，`formal_frozen_directional_count.yaml`
用于正式评估。两者保持相同模型、表示和数据根目录；阶段字段用于避免误用配置。
执行路径默认相对项目根目录，命令从该目录运行。

## 数据准备

数据与权重未包含在本项目中。数据来源和使用权限遵循原始公开数据集说明。
只使用干净 Fiber/Starlink 数据。现成分区文件应放在：

```text
npz_dataset/
├── tor_fiber/
│   ├── train.npz
│   ├── valid.npz
│   └── test.npz
└── tor_starlink/
    ├── train.npz
    ├── valid.npz
    └── test.npz
```

NPZ 包含 `X`、`y` 和 `sample_ids`。`X` 为 `(N,10000,2)` 的补零事件数组，
两列为相对时间和带符号方向标记；`y` 为 0–74 的类别标签。
训练/验证/评估各为 4,860/540/600 项，两个域的同一访问保持相同划分角色。

已有固定分区时，生成计数缓存：

```bash
python data_process/prepare_datasets.py --skip-convert --skip-split --splits train valid test
```

输入内容哈希由计数缓存和主流程记录。

若从原始文本轨迹开始，将文件放入 `dataset/tor_fiber/` 和
`dataset/tor_starlink/`，文件名为 `<类别>-<访问编号>`，每行两列、制表符分隔。
先用下列命令查看计划，再手动执行转换和划分：

```bash
python data_process/prepare_datasets.py --splits train valid test --dry-run
python data_process/prepare_datasets.py --splits train valid test
```

复现论文结果时应使用相同的固定分区及样本顺序。

## 运行

只查看 6 次训练、12 项评估的计划，不执行训练：

```bash
python scripts/run_main_experiment.py --dry-run
```

完整主试验，由操作者手动执行：

```bash
python scripts/run_main_experiment.py --gpus 0
```

流程先完成两域三种子训练，每个模型按源域验证 Macro-F1 选取权重，随后运行
F→F、S→S、F→S、S→F。运行ID自动生成，输出分别位于 `logs/`、`checkpoints/`
和 `results/`。续跑使用日志中的原运行ID：

```bash
python scripts/run_main_experiment.py --gpus 0 --resume RUN_ID
```

单独训练、评估示例：

```bash
python scripts/train.py --dataset tor_fiber --seed 42 --run-id manual_fiber42
python scripts/test.py \
  --config configs/formal_frozen_directional_count.yaml \
  --dataset tor_starlink --source-dataset tor_fiber --seed 42 \
  --checkpoint checkpoints/directional_count/candidate/manual_fiber42/tor_fiber/seed_42/best_model.pth \
  --output results/manual_fiber42/F_to_S_seed_42.json
```

模型输入为 120 秒、2×2700 方向计数，前 5,000 个有效事件；100 epochs、batch 64。
输出包含逐项指标、预测及标签；`F1-score` 字段对应 Macro-F1，指标为百分制。
三种子汇总使用样本标准差 `ddof=1`。


许可状态见 `LICENSE_NOTICE.md`。
