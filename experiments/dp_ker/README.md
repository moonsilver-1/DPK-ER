# DP-KER Lab

本项目已经整理为 `uv` 管理的标准 Python 项目。请不要再在全局 Python 环境里直接 `pip install`，即使全局环境之前装乱了，也不会影响这个 `uv` 项目。

## 环境

Windows PowerShell 安装 `uv`:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

初始化环境:

```powershell
uv sync
```

建议的可选依赖:

```powershell
uv sync --extra sentence-transformers
```

如果 `sentence-transformers` 装不上，实验代码会自动回退到 `TF-IDF + SVD` embedding，pipeline 仍然可以跑通。

## 数据

原始 Hugging Face 数据放在:

```text
data/raw/resume-score-details/
```

转换后的实验数据放在:

```text
experiments/dp_ker/data/
```

结果输出放在:

```text
experiments/dp_ker/results/
```

## 下载数据

```powershell
uv run huggingface-cli download netsol/resume-score-details --repo-type dataset --local-dir data/raw/resume-score-details
```

如果 Hugging Face 连接较慢，先执行:

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

然后再运行下载命令。

## 运行脚本

```powershell
uv run python experiments/dp_ker/src/inspect_dataset.py
uv run python experiments/dp_ker/src/convert_hf_dataset.py
uv run python experiments/dp_ker/src/run_experiment.py
uv run python experiments/dp_ker/src/run_ablation.py
uv run python experiments/dp_ker/src/run_privacy_budget.py
uv run python experiments/dp_ker/src/run_all_labs.py
```

## 约定

- 本项目使用 `uv` 管理环境，不建议使用全局 Python 环境。
- 如果你之前在全局环境里装过很多包，也不用清理，`uv` 项目会自己管理隔离环境。
- `toy dataset` 只能用于 pipeline 测试，不能写入 `results/*.csv`。
- 论文主结果必须来自 `data/raw/resume-score-details/` 转换后的公开数据集，且必须由实际脚本运行生成。
- 所有随机数 `seed` 固定为 `3407`。

## CSV 输出与论文表格

- `experiments/dp_ker/results/main_results.csv` 对应 `Main Results`。
- `experiments/dp_ker/results/ablation_results.csv` 对应 `Ablation Study`。
- `experiments/dp_ker/results/privacy_budget_results.csv` 对应隐私预算分析表。
- `experiments/dp_ker/results/dataset_statistics.csv` 对应数据集统计表。

如果你的论文最终表号不同，可以在这里直接替换表号说明。
