# KoopCE

KoopCE 是一个围绕 Koopman coarse-graining、空气质量、Kuramoto 和 Rulkov map 等实验整理的研究代码仓库。当前仓库以轻量结构保存通用工具、数据文件、实验脚本与 notebook。

## 当前目录结构

```text
koopCEgis/
├── data/                    # 数据文件与数据生成/仿真函数
│   ├── base.py
│   ├── data_func.py
│   ├── dataset_bthsa.nc
│   ├── dataset_yrd.nc
│   ├── stations_bthsa.csv
│   └── stations_yrd.csv
├── exp/                     # 各实验分支
│   ├── air_quality/
│   ├── analysitic/
│   ├── kuramoto/
│   └── rulkov_map/
├── doc/                     # 研究文档与参考资料
├── tools.py                 # 主要分析、Koopman 拟合与绘图工具函数
├── requirements.txt
├── LICENSE
└── README.md
```

## 主要代码入口

- `tools.py`
  - 仓库的核心工具模块，包含 Koopman operator 拟合、空气质量数据处理、Rulkov map 指标分析等通用函数。
- `data/data_func.py`
  - 生成 Kuramoto、多群神经元等仿真数据，并提供部分绘图辅助函数。
- `exp/air_quality/run_air_macro_sweep.py`
  - 空气质量宏观分析 sweep 脚本。
- `exp/kuramoto/run_whitened_macro_sweep.py`
  - Kuramoto whitening / macro sweep 脚本。
- `exp/rulkov_map/run_map_analysis_sweep.py`
  - Rulkov map 状态分析 sweep 脚本。

## 数据与实验说明

- `data/dataset_yrd.nc` 与 `data/stations_yrd.csv`
  - 长三角空气质量数据及站点元数据。
- `data/dataset_bthsa.nc` 与 `data/stations_bthsa.csv`
  - 京津冀及周边空气质量数据及站点元数据。
- `exp/air_quality/`
  - 空气质量 notebook、分析脚本与调参材料。
- `exp/kuramoto/`
  - Kuramoto 双群体、频谱与 coarse-graining 相关实验。
- `exp/rulkov_map/`
  - Rulkov map / 两群神经元同步态分析实验。
- `exp/analysitic/`
  - 解析型或理论推导相关 notebook。

## 运行示例

在仓库根目录安装依赖后，可以直接运行：

```bash
python exp/air_quality/run_air_macro_sweep.py --help
python exp/kuramoto/run_whitened_macro_sweep.py --help
python exp/rulkov_map/run_map_analysis_sweep.py --help
```

如果你主要想找 notebook：

- 空气质量实验在 `exp/air_quality/`
- Kuramoto 实验在 `exp/kuramoto/`
- Rulkov map 实验在 `exp/rulkov_map/`
- 理论分析实验在 `exp/analysitic/`

## 依赖说明

依赖列表见 `requirements.txt`。其中 `pysindy`、`pykoop`、`xarray`、`cartopy` 等包用于动态系统建模、数据加载和绘图分析；如果只运行部分 notebook，可按需裁剪环境。
