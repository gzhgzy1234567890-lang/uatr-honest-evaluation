# W1–W4 操作手册

> 项目：DeepShip 水下声目标识别，严格评估协议基准与域偏移归因
> 目标期刊：Applied Acoustics（中科院大类 2 区 / JCR Q1）
> 约束：仅 Kaggle 免费额度（30 GPU h/周，单会话 ≤12 h，`/kaggle/working` 20 GB）

---

## ⭐⭐ 论文方向定稿（2026-10-06）：审计/测量文（路线 A）——方法路线已全部被零成本探针否证

**🔴 2026-10-06 晚终局：W3-H 全量 NO-GO，五次干预全部关闭，进入写作。**

| 干预 | 判别实验 | 结果 |
|---|---|---|
| ① 容量 | ResNet18 11.2M（`rc2`） | 训练集记到 0.9999，测试仍 0.3777 |
| ② 损失再平衡 | 离线 logit adjustment 天花板 | macro-F1 上限 **+2.0 pp** |
| ③ 选择性弃权 | accuracy–coverage 曲线 | 平坦：acc@50% 仅 0.367；head-only 对尾部弃权率 2.7% |
| ④ 身份数加权 | 剂量反应（召回 vs 船数/片段数） | 召回对片段数跟踪更好；Passenger 42 船召回 0.000 → 前提不成立 |
| ⑤ 预训练表征 | 冻结 AST + 线性探针（W3-H 全量） | **尾部 top-2 0.114 → 0.116，一模一样**；acc 反而 0.3183→0.2677 |

⇒ **坍缩是语料内在性质**。90M 监督 CNN 与 90M AudioSet 预训练 transformer 的尾部可解码性相同。
**论文升级**：不是"我们量了一下"，而是"领域会提出的全部干预都被逐一实测否定"——
"Neither capacity, nor pretraining, nor loss reweighting rescues the tail: what UATR benchmarks
actually measure under honest evaluation"。

**🟢 2026-10-07：W3-I 分层验证 CONFIRMED（18 runs / 4.87 h）——remedy 章节闭环**

| allocation | n | mean | observed sd | **pred.sd（纯元数据）** |
|---|---|---|---|---|
| random(rec2) | 5 | 0.3157 | 4.69 pp | - |
| **clipmass** | 6 | **0.3215** | **1.24 pp**（约等于初始化地板） | 1.12 |
| countstrat | 6 | 0.3695 | 4.85 pp | **4.85（精确命中）** |
| sgkf | 6 | 0.3093 | 2.60 pp | 2.28 |

sd ratio **0.26**（门槛 0.6）；分辨 2 pp 所需抽样 **86 -> 6**。三个方案的 pred.sd 全部命中
-> **只用元数据就能在训练前预测真实模型的 draw 方差**（核心主张的样本外验证）。
countstrat 双重失败：方差不降 + 均值虚高 +4.8 pp；sgkf 只把方差减半——"标准工具不是最优"有定量答案。
**立刻把 w3i-runs + w3i_stratified.json 存成 Kaggle Dataset。**

**🟢 2026-10-07：W3-J fine-tuned AST = PARTIAL——六次干预链关闭，进入写作**

| | acc | macro-F1 | 尾部 top-2 |
|---|---|---|---|
| cnn_small | 0.3183 | 0.108 | 0.114 |
| frozen AST | 0.2677 | 0.112 | 0.116 |
| **fine-tuned AST** | 0.2584 | **0.142** | **0.178** |

PARTIAL（预注册 0.30/0.15）：微调是**唯一让 macro-F1 动的干预（+3.4 pp）**，但尾部 top-2 远低于 GO 线。
逐类三层结构：救活（Fishing .296->.442、PleasC .208->.339）/ 不动（Passenger）/ 死（SAR、Tanker），
头部反而下降（Cargo recall .584->.322）。val 在 ep2 见顶 -> 基础模型 2 epoch 即过拟合。
**新章节：有效类结构的三层图景（可学 / 可部分救活 / 不可救），零额外实验。**
**实验链关闭。剩余可选：泛化保险（~4 h）、w3e_aggregate（CPU）。下一步 = 写作。**

**🟢 2026-10-07 深夜：W3-K 泛化 CONFIRMED（logreg 0.34 / cnn_wide 0.36，双双 <= 0.6）**
容量阶梯三个点全部 sd 收缩（logreg 2.76->0.94、cnn_small 4.69->1.24、cnn_wide 3.07->1.11 pp）；
构成预测器战绩 **5/5**（pred.sd 全部命中区间）。
**🔴 cnn_wide clipmass 仅 3/6 draws（预算切断）——重跑脚本补齐 seeds 3-5（约 1 h，去重自动续）再存 Dataset。**

**🟢🟢 2026-10-08：W3-K2 收官——cnn_wide 补齐（6-draw sd 约 1.36 pp，比值约 0.44 仍泛化）+ DeepShip 无操作对照 CONFIRMED（pred.sd 0.30 pp、比值 1.33 双过）。**
**构成预测器的完整故事：OceanShip 说"要分层"（85% 构成方差），DeepShip 说"分层没用"（构成仅 ~10%）——两次都对。**
**⏳ 用户：合并两份 results.csv 重跑 w3k_gencheck 分析段取 cnn_wide 精确 sd；两个新输出 Create Dataset。**
**实验链 100% 完成 → 写作（IEEE JOE）。**

**🟢 显著性检验已补（零成本）**：clipmass vs random 方差比 **14.40，F-test p=.0119、置换 p=.0021** -> 收缩显著；
sgkf 比值 3.25 不显著（p=.085）；countstrat 与随机不可区分（比值 0.93）。
**⚠️ countstrat 均值偏置 +5.4 pp 的 Welch p=.096 —— 论文里只能写 "observed upward shift"，不能写 confirmed。**


**剩余实验（封审稿人的洞，共 ~16 GPU h）**：
1. **fine-tuned AST** × 3 seed × vessel_disjoint（~5 h）——封死"你只测了冻结"这最后一个洞；
2. **W3-G 解耦网格** 3 协议 × 10 抽样 × 2 初始化（~10 h）——方差分解样本外验证 + 构成预测器；
   **前置：把本地 10-seed 划分文件刷进 Kaggle 数据集**；
3. `w3e_aggregate`（CPU）——bootstrap CI；4. 可选 ShipsEar 第三语料（~1.5 h）。

### 论文核心（全部已有数据支撑）

**标题方向**："What do underwater-acoustic benchmarks actually measure? An audit of split protocols,
variance, and the effective class structure of OceanShip and DeepShip"

1. **协议阶梯 + 方差分解**：三级划分（`clip_random`/`rec_random`/`vessel_disjoint`）× 双表征 × 5 seed（`rec2`，33 配置）；
   同协议名字 OceanShip 只值 +1.6 pp、DeepShip 值 +30.4 pp；`vessel_disjoint` 的 4.69 pp 方差中 **85% 由测试集类别构成解释**（纯元数据、训练前可算）；
   配对设计省 9.4 倍算力；required-N 表（分辨 2 pp：非配对 87 抽样 / 配对 9）。
2. **有效类结构**：诚实协议下 OceanShip 12 类坍缩为 3 类（混淆矩阵：小艇组 70–80% → Towing，Tanker 60% → Tug，Tug 37.8% → Cargo）；
   尾部 top-2 ≈ 机会（0.114）且泄漏协议下也不可恢复（语料性质）；重加权天花板 +2.0 pp；置信度不跟踪正确性（曲线平坦）；
   **DeepShip 对照：不坍缩**（4/4 类召回 55–75%）→ 坍缩是类粒度性质，不是模型/协议伪影；身份地板（<5 船类不可学，预测命中）。
3. **泄漏的两面**：存在性（元数据天花板 99.44%）≠ 可利用性（1-NN 同录音 5.68% = 180× 随机但绝对量小）；
   一个标量 78.18% 的检索下界；切片策略（首尾相接 vs 任意时刻）决定泄漏大小与诚实结果的方差——同一枚硬币。

### 剩余实验（按性价比）

| 项 | 成本 | 目的 | 状态 |
|---|---|---|---|
| **W3-G 解耦网格**：3 协议 × 10 抽样 × 2 初始化（OceanShip）| ~10 GPU h | 方差分解的样本外验证 + **构成预测器**（元数据预测未见抽样的分数）| 待建 notebook；**先刷 10-seed 划分进 Kaggle 数据集**（本地 `out/` 已就绪）|
| label shuffling 对照 × 3 协议 | ~0.5 h | 排除"提升来自 bug"的质疑 | 待加进 W3-G |
| ShipsEar 第三语料（recording 级 = 文件级，2,223 片段） | ~1.5 h | 泛化性第三点 | 可选 |
| `w3e_aggregate`（录音簇 bootstrap CI） | CPU | Table 1/2 误差棒 | 用户已存 W3_data 可直接跑 |

### 用户手头动作

1. **把本地 `out/oceanship_splits.csv.gz` 与 `out/splits/splits.csv.gz`（均已 10 seed，0–4 逐字节不变）刷进 Kaggle 数据集 `under-water-features`**；
2. W3_data（W3-A 输出）已存 ✅——`w3e_aggregate` 现在就能跑。

---

## ⚡ W2 一条命令跑完（2026-09-29 起用这个）

**用 `notebooks/kaggle_build_feature_dataset.ipynb`**，它替代了下面所有手工步骤。

它的作用：一次性给三个语料库（Oceanship / DeepShip / ShipsEar）生成同一套冻结特征，
产物约 **14 GB**，之后所有建模 notebook 只挂这一个数据集。

流程：

1. Kaggle → New Notebook → 上传 `notebooks/kaggle_build_feature_dataset.ipynb`
2. Add Input 挂三个数据集：`deepship-raw`、`oceanship`、`workspace`
3. Accelerator **不需要 GPU**（`--device cpu`，故意的：冻结产物不应依赖当天分到哪块卡）
4. **Save Version → Save & Run All (Commit)** 后台跑。**实测 1859.6 s（31 分钟）**：
   探针 ~40 s（采样 3,000）／生成划分 ~10 s／测量 ~5.5 min／写入 ~23 min
   （Oceanship 16.0 min @56 clip/s、DeepShip 6.8、ShipsEar 0.5）
5. **构建是两阶段的**：先测完三个语料库 → **一次空间检查** → 才开始写。
   所以空间不足时**一个字都不会写**，重跑是干净的。mel+spec = **14.41 GB**，
   `/kaggle/working` 20.9 GB 够用。
6. **同一 session 里重跑**要清掉上一次的阵列（构建 cell 会打印总量与命令），
   或把 cell 顶部 `PREVIOUS = True`。
7. **cell 4 会现场重建划分并逐内容比对**（见下），所以 **Kaggle 上那份 workspace
   数据集不需要重传**——即使里面是旧的、不可复现的划分，产物也会是生成器产出的。
8. 跑完后打开该版本 → **Output 标签页 → Create Dataset** → 选中 `uwatr-features`
   （这是 Kaggle 官方路径，见 [docs/datasets](https://www.kaggle.com/docs/datasets)：
   "click Create Dataset from the Output tab on your rendered Notebook"——
   **不需要先下载再上传**）

### 🔴 冻结划分会被重建并校验，不是"相信挂载的那份"

cell 4 会现场跑两个生成器，并把结果与挂载文件**按内容**比对（gzip 头让字节数不可信）：

| 生成器 | 输入 | 产出 |
|---|---|---|
| `w1_04_oceanship_inventory.py` | `metadata/oceanship_fg_{train,test}.csv` | inventory + Oceanship 划分 + 泄漏审计 |
| `w1_02_make_splits.py` | `deepship/deepship_recordings.csv` | DeepShip 划分 + 摘要 |

- 一致 → 保留挂载的那份，打印 `verified reproducible`。
- **不一致 → 换成生成器的产出，并打印差多少行**。这一条是有来历的：`w1_04` 曾经用内置
  `hash()` 做种子（受 `PYTHONHASHSEED` 加盐），同一份 CSV 每次跑出不同的"冻结划分"。

两个语料库的**协议名不同**，不要混用：Oceanship 是 `clip_random / rec_random /
vessel_disjoint`；DeepShip 是 `clip_random / rec_random / temporal`（**没有 vessel_disjoint**）。
验证 cell 现在会把每一份划分里**实际存在的**协议与每个 seed 都打印出来。

**先看第 4 个 cell 的 `inputs` 与 `DEGRADED` 两行**（v3 起）：

- `DEGRADED = []` → 三个语料库都会建。
- `DEGRADED = ['ShipsEar']` → ShipsEar 没找到，只跳过它，**其余照建**；
  cell 会把所挂 `workspace` 数据集里的**全部文件与目录**（带尺寸）打出来，照着找原因。
- 缺 `oceanship inventory/splits` 但官方 `oceanship_fg_*.csv` 在 → cell 会**现场重建**
  （同一个生成器，`sha256` 种子，产物与冻结版逐字节一致）。
- 只有既无派生文件又无 CSV 才停。数据集名不一样就填第 1 个 cell 的 `MANUAL`：
  `MANUAL["workspace"] = "/kaggle/input/datasets/<owner>/<slug>"`。

**🔴 Kaggle 会在创建数据集时自动展开归档**，所以"文件找不到"几乎都是形态变了、不是丢了：

| 你上传的 | Kaggle 上实际是 |
|---|---|
| `uwatr-workspace.zip` | `workspace/uwatr-workspace/…`（多一层） |
| `3090latest_wav.zip` | `oceanship/3090latest_wav/` 53,770 个散装 wav |
| `ShipsEar.zip` | `shipsear/ShipsEar/shipsear_5s_16k/<class>/<group>/*.wav` |
| `oceanship_inventory.csv.gz` | `metadata/oceanship_inventory.csv`（**连 `.gz` 也展开**） |

notebook v4 与构建器**两种形态都能吃**：按魔数嗅探读 `.gz`（不看扩展名）；ShipsEar 从归档或
解压树都能读，且两种形态产出的 `clip_id` **逐字相同**；产物一律恢复成文档里的名字与压缩。

notebook 同时嵌入构建器与 `w1_04_oceanship_inventory.py`（都逐字节一致），
写到 `/kaggle/working/scripts/`，**不依赖所挂 workspace 里的脚本版本**。

产物结构：

```
uwatr-features/
  build_features.py         产生这份缓存的脚本本体
  manifest.json             行数 / 形状 / 配置 / 逐 split 行数
  README.md                 表意、连接方式、挂载方式
  oceanship/{logmel.npy, spec.npy, index.csv}
  deepship/{logmel.npy, spec.npy, index.csv}
  shipsear/{logmel.npy, spec.npy, index.csv}
  splits/                   3 协议 × 5 seed 冻结划分 + 泄漏审计 CSV
```

表示（三个语料库完全一致）：16 kHz 单声道、5 秒 / 80,000 采样、
log-mel 与线性谱各 (N,128,313) fp16、dB 相对每条片段峰值、下限 −80 dB。

**与建模的连接键**：Oceanship 用 `clip_id`，DeepShip 用 `(rec_key, clip_idx)`，
对应 `splits/` 里同名划分文件。`index.csv` 还带 `mmsi` / `label` / `src_dur_s`，
所以域划分和时长消融都是一次 join。

> 为什么短片段是**补齐而不是丢弃**：见 `docs/百度分卷处理.md` §四——
> 时长在 Oceanship 里与类别相关（Passenger 中位 3.67 s vs Tug 5.00 s），
> 丢弃会静默砍掉 1/8 语料并破坏类别平衡；补齐则引入"静音比例"这条捷径，
> 所以真实时长逐条写进 `index.csv` 的 `src_dur_s`，消融随时可做。

---

## 当前状态

**W1 已经完成并在本地验证过。** 你可以直接进入 W2。

| 阶段 | 状态 | 说明 |
|---|---|---|
| W1 元数据 + 划分 + 泄漏审计 | ✅ **已完成** | 产物在 `out/`，纯 CPU，不需要 GPU |
| W2 特征缓存 | ⏳ 待你在 Kaggle 跑 | 约 30 分钟，一次性 |
| W3–W4 基准实验 | ⏳ 待你在 Kaggle 跑 | 约 13 GPU h，分 2–3 个会话 |

**W1 已经拿到的结果（论文的核心论据）**

```
609 条录音 / 47.14 h / 33,710 个 5 秒片段      (官方口径 47h04m / 33,693，吻合)
592 条录音带元数据 (97.2%)  ← 但只有 561 条 (92.1%) 的元数据是可信的，见下
248 条录音来自"被记录超过一次"的船 → 天然的同标签跨域配对

泄漏审计：
  clip_random  99.9% 的测试片段来自模型训练过的录音
  rec_random    0.0%
  temporal      0.0%
```

`clip_random` 那 99.9% 就是整篇论文的引子。任何在这个协议下报出的 99% 精度，测的其实是"模型还记不记得这条录音"，不是"能不能认船型"。

---

## 目录结构

```
G:\Work_work\under_water\
  src\                        可执行脚本（本机与 Kaggle 通用）
    w1_01_build_metadata.py     录音级元数据表（含多策略 join + 时长校验）
    w1_02_make_splits.py        三种协议 × 5 组划分
    w1_03_audit_leakage.py      泄漏审计（论文的核心数字）
    w2_01_cache_features.py     特征缓存（logmel + linear spec）
    w3_00_sweep.py              sweep 驱动（断点续跑、时间预算）
    w3_01_run_benchmark.py      单个配置训练与评测
    w3_02_aggregate.py          录音级 bootstrap CI + 论文表格
  data\metadata\              DeepShip 清单 + 4 个官方 metafile（已下载）
  out\                        W1 产物（待上传 Kaggle）
    deepship_recordings.csv
    splits\splits.csv.gz
    w1_0*_report.txt
```

---

## 一个必须先处理的前置事项：ShipsEar

**ShipsEar 全文数据库需要发邮件申请，不能直接下载。**

官方页面（`atlanttic.uvigo.es/underwaternoise/`）只提供 `samples.zip` 样本，全库要联系作者。

**现在就发这封邮件**（越早越好，可能要等几天到几周）：

> 收件人：ShipsEar 作者（页面底部联系方式）
> 主题：Request for access to the ShipsEar database
> 正文：简要说明你是某某单位的研究人员，正在做水下声目标识别的评估协议研究，希望获取完整数据库用于学术研究，承诺仅用于非商业研究并正确引用 Santos-Domínguez et al. 2016。

**好消息**：**这不阻塞 W1–W4。** DeepShip 单独就能支撑三种协议的全部基准实验。ShipsEar 是用在后续的跨数据集部分（论文第 6 节）和 provenance 重建（C1）。

**备选**：HuggingFace 上有 `peng7554/DS3500`（CC-BY-4.0，可自由下载），内含 ShipsEar 的 16 kHz / 5 秒分段版本（1948 段 / 5 类 A–E）**加上**基于 BELLHOP 的深海增强版，还带距离标签（1–11 km）和深度标签。可以直接用来做跨域实验的备选目标域。缺点是它已经切成片段了，丢失了录音级 provenance，所以**不能替代**原始 ShipsEar 的 C1 用途。

---

## W2：特征缓存（约 30 分钟，一次性）

### 1. 上传 W1 产物 + 脚本为 Kaggle Dataset

把下面这些东西打包成一个 Dataset（总共约 1.6 MB），命名 `uwatr-protocol-v1`：

```
uwatr-protocol-v1/
  deepship_recordings.csv        <- out/deepship_recordings.csv
  splits/
    splits.csv.gz                <- out/splits/splits.csv.gz
    split_summary.csv
  reports/                       <- out/w1_0*_report.txt（可选，便于核对）
  src/                           <- src/ 下全部 7 个 .py
```

脚本和数据放同一个 Dataset 是最省事的做法，所有 notebook 都能直接调用，不用重复粘贴。

### 2. 新建 Notebook

- Accelerator：**GPU P100**（这一步其实 CPU 也够，但 GPU 快）
- Internet：不用开
- Attach：
  - `tangqiji/deepship-raw`（公开数据集，直接搜）
  - `uwatr-protocol-v1`（刚上传的）

### 3. 冒烟测试（先跑这个）

确认数据挂载正确、脚本能跑通：

```python
!pip install -q soundfile scipy
!python /kaggle/input/uwatr-protocol-v1/src/w2_01_cache_features.py \
    --root /kaggle/input/deepship-raw \
    --recordings /kaggle/input/uwatr-protocol-v1/deepship_recordings.csv \
    --out /kaggle/working/smoke --limit 5
```

看到 `[plan]` 与 `rows=` 递增就说明一切正常。

### 4. 正式运行

```python
!python /kaggle/input/uwatr-protocol-v1/src/w2_01_cache_features.py \
    --root  /kaggle/input/deepship-raw \
    --recordings /kaggle/input/uwatr-protocol-v1/deepship_recordings.csv \
    --out /kaggle/working/cache
```

### 4. 固化

用 **Save Version → Save & Run All (Commit)** 后台跑（约 30 分钟，不占交互会话）。

跑完后在输出页面 **Create Dataset**，命名 `uwatr-logmel`。产物约 5.4 GB，之后所有训练直接挂载，不再重算。

---

## W3–W6：六个 notebook，按顺序各跑一个会话

**实验设计与对标清单见 `docs/实验计划-复现对标与问题清单.md`**（复现谁、对标哪一行、要分析出哪些问题）。

前置：把 W2 的 `uwatr-features` 存成 Kaggle 数据集 **`under_water_features`**（实测挂载路径
`/kaggle/input/datasets/gzhwyf/under-water-features/uwatr-features`），其余 notebook 都只挂它。

| 顺序 | Notebook | 内容 | 加速器 | 预计 |
|---|---|---|---|---|
| **0** | **`kaggle_w3a3_leak_probe.ipynb`** | ✅ **已跑（v1）**；**v2 改成两个语料**：泄漏在声学上到底有多可利用 | **CPU** | ~5 min |
| **1** | `kaggle_w3a4_deepship_reference.ipynb` | ✅ **已跑**（tag `ds1`）：DeepShip 标定，对标文献 72.27% | GPU | 2.3 h |
| **2** | `kaggle_w3a2_recipe_check.ipynb` | ✅ **已跑**（tag `rc2`）：模型能不能拟合训练集 | GPU | 2.7 h |
| **3** | `kaggle_w3a_headline.ipynb` | ✅ **已跑**（tag `rec2`，33 配置，9.25 h）：主表 Table 2/3 | GPU | 9.25 h |
| 4 | `kaggle_w3b_baselines_scale.ipynb` | 基线与规模：logreg / cnn_tiny / cnn_wide / resnet18 / efficientb0 = 37 配置 | GPU | 5–7 h |
| 5 | `kaggle_w3c_diagnostics.ipynb` | 诊断：recording-id NMI、域可分性、时长捷径、label shuffling 对照 | GPU | 2–4 h |
| 6 | `kaggle_w3d_domain_generalization.ipynb` | C4：IRM / VREx / DANN × 4 种域定义（含负对照）+ 跨语料零样本 = 22 配置 | GPU | 4–6 h |
| 7 | `kaggle_w3e_aggregate.ipynb` | 全部表格与图（录音级 bootstrap CI） | **不需要 GPU** | 分钟级 |

> **`rec1` / `rc` 两批结果全部作废**（学习率调度从未推进）。**当前有效批次 = `rc2` / `ds1`。**
> **下一步先跑 W3-A3 v2**（CPU、几分钟、**零 GPU 预算**）：它把"泄漏有多大"从元数据一路量到声学，
> 并解释为什么同一个协议在 DeepShip 上吃掉 30 个点、在 OceanShip 上只吃掉 7 个点。

### 🔴 核心发现：同一个 `clip_random`，两个语料差了整整一个量级

`src/w1_05_recording_structure.py`（**零成本，只读 out/ 的元数据与冻结划分，不碰音频**）已经算出：
同一个 `clip_random` 协议，在两个语料上交给模型的东西差了整整一个量级。

| | DeepShip | OceanShip |
|---|---|---|
| recordings | 609 | 10,168 |
| 片段/录音（中位） | **51** | **3** |
| `clip_idx` 恰为 0..N-1 | **609/609** | 无此索引 |
| (N × 5 s) / 录音时长 | **0.994** | n/a |
| 测试片段在训练集里"有相邻片段" | **89.6%** | 5.68%（实测） |

**DeepShip 把每条长录音切成首尾相接的 5 s 窗口**，所以第 i 段和第 i+1 段是**同一段音频**；
**OceanShip 从连续水听器流里任意时刻切出带标注的短片段**，同一录音的两个片段彼此无关。
→ "用了随机划分"**不是一句可比的规格**，而这件事没有任何数据集论文说明过。这就是本文的题眼。

### 🔴 `rec1` / `rc` 的全部结果作废 —— 学习率调度从未被推进

2026-09-30 的两次 Kaggle 运行（W3-A 32 配置 6.1 h；W3-A2 7 配置 3.6 h）都**跑通但无效**，根因是
`w3_run.py` 里 `LambdaLR` 构造了却**从未 `step()`**：

```python
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)   # ← 建了
for ep in ...:
    loss = train_one_epoch(...)                          # ← 但从没 sched.step()
```

于是学习率永远是 `lr_at(0)`。带 warm-up 时 `lr_at(0) = 1/(warmup × steps_per_epoch)`：

| 配置 | 请求 lr | **实际 lr** | train acc |
|---|---|---|---|
| `rc_aff24` / `rc_aff60` / `rc_ds60`（warm-up 3，588 步/轮） | 1e-3 | **5.7e-7** | 0.273 / 0.288 / 0.286 |
| `rc_res30`（warm-up 3） | 5e-4 | **2.8e-7** | 0.303 |
| `rc_base24`（**warm-up 0**） | 3e-4 | **3e-4** ← 唯一真在训练 | **0.359**（最高） |

**"容量无用、轮数无用、归一化反而更差"这个模式，正是"网络没离开初始化"的样子。** W3-A2 由此得出的
"该重新考虑表征"的结论是错的。

**已修（`src/w3_run.py`）**：调度器每个优化步 `step()`；每轮打印实时 `lr=`；
`results.csv` 新增 `lr_last`；**调度不动现在是硬错误**；`--smoke` 自动把 warm-up 设 0。
**判据**：正常余弦跑完 `lr_last ≈ 0.00e+00`，卡住的会是 1e-7 量级。

### ✅ `rc2` 配方体检已跑完（2026-10-01，9660 s / 6 配置 / 无报错）

| config | K | lr set | ep | **train** | **test** | gap | predK |
|---|---|---|---|---|---|---|---|
| `ovf512`（512 片段） | 12 | 3e-3 | 200 | **0.9922** | 0.2533 | +0.739 | 10 |
| `f24_1e3` | 12 | 1e-3 | 24 | 0.3543 | 0.3507 | +0.004 | 5 |
| `f24_3e3` | 12 | 3e-3 | 24 | 0.4011 | 0.3649 | +0.036 | 8 |
| `f60_3e3` | 12 | 3e-3 | 60 | 0.5157 | **0.3679** | +0.148 | 9 |
| `res30`（resnet18 11.2M, pretrained） | 12 | 1e-3 | 30 | **0.9999** | **0.3777** | +0.622 | 12 |
| `v60_3e3`（vessel_disjoint） | 12 | 3e-3 | 60 | 0.5084 | **0.2722** | +0.236 | 9 |

全部 `lr_last = 0.00e+00`，调度确实走完。四个结论：

1. **流程没问题**：`ovf512` 把 512 片段记到 0.9922；resnet18 把**整个 37,626 训练集**记到 **0.9999**
   并预测出全部 12 类。**欠拟合问题结束。**
2. **诚实协议落差存在**：同一配方（cnn_small / 60 轮 / lr 3e-3）`clip_random` **0.3679** vs
   `vessel_disjoint` **0.2722** → **9.6 pp**。这是论文要的那个数。
3. **参数量无用**：pretrained resnet18 11.2M 只比 136,692 参数的 cnn_small 高 **+1.0 pp**
   （0.3679 → 0.3777）→ **直接复现了 Soti et al.「11.17M ResNet18 无提升」**。
4. **天花板远未达到**：同一划分上，只预测"该片段所属船在训练集里的多数标签"（**完全不用音频**）
   得 **99.44%**。模型只有 0.38。

→ **新的待解问题不是"模型能不能学"，而是"泄漏所依赖的录音身份，到底在不在特征里"。**
如果一条录音的两个片段在声学上并不相关，那么这个 clip 级泄漏就是**纯元数据的**，
任何音频模型都到不了 99%——**而这本身就是一个结果**（泄漏论证缺的另一半），不是 bug。

### ✅ W3-A3 已跑（CPU，82.6 s）——泄漏是**纯元数据的**

| 测的东西 | 结果 | 怎么读 |
|---|---|---|
| 元数据天花板（`clip_random` s0，预测船的多数训练标签，**不用音频**） | **99.44%** | 划分单元值多少 |
| **1-NN：最近邻是否来自同一录音** | **5.68%** | 随机基线 **0.0315%** → **180 倍**，身份**在**特征里 |
| 1-NN：最近邻是否同一船 | 6.51% | 随机基线 0.6232% → 10.4 倍 |
| 同一录音 vs 不同录音距离比 | **0.898** / P(更近)=0.557 | 弱但非零 |
| 1-NN 标签正确率 | 25.48% | 最好常数类 = 22.05% → **只高 3.4 pp** |
| 256 维岭回归线性探针 | 27.44% | |
| 只用 `src_dur_s` 的 1-NN / 时长分箱多数标签 | 9.34% / **22.81%** | 时长几乎就是常数基线 |
| 类均值 | 3.71%（**低于**随机） | 类别均值几乎共线 |

**结论**：录音身份**确实在特征里**（180× 随机），但**弱而弥散**——一条 5 s 片段无法可靠地指认自己的
录音。**99.44% 那个天花板主要是元数据的，不是声学的**，所以任何音频模型都到不了它。
**这本身是论文缺的另一半**，必须写进去：泄漏的"可利用性"和"存在性"是两件事。

**同时留下一个必须回答的问题**：所有廉价探针都在 22–27% 一档（CNN 36.79%），比文献的 72.27%
低 40 个点。要么流水线弱，要么 12 类本来就比 4 类难得多——**没人说过这件事**。

### ✅ W3-A4 已跑（tag `ds1`，8134 s / 11 配置 / 无报错 / `lr_last` 全 0）

| corpus | 协议 | seed 0/1/2 | 均值 | 训练精度 | 该协议的元数据天花板 |
|---|---|---|---|---|---|
| **DeepShip**（4 类） | `clip_random` | 0.9179 / 0.9288 / 0.9191 | **0.9219** | 0.97 | 船 97.53% ｜ 录音 **99.82%** |
| DeepShip | `rec_random` | 0.6598 / 0.7167 / 0.6761 | **0.6842** | 0.98 | 录音 0.00% ｜ 船 54.91% |
| **DeepShip** | `temporal` | 0.6228 / 0.6244 / 0.6075 | **0.6182** | 0.98 | 录音 0.00% ｜ 船 **55.14%** |
| OceanShip（12 类） | `clip_random` | 0.3649 | | 0.40 | 船 99.45% ｜ 录音 90.80% |
| OceanShip | `vessel_disjoint` | 0.2926 | | 0.43 | 船 0.00% ｜ 录音 0.00% |

**判读（注意：判定行写的是 "well below the literature"，但那是对 0.7227 这个单一数字说的）**

1. **流水线不是坏的**：DeepShip 上同配方到 **0.6182**（诚实协议）与 **0.9219**（泄漏协议）。
   文献 72.27% 与我们差 **10.4 pp**，而我们对 DeepShip 的**整条流水线**（小 CNN / 24 轮 / 128 维 log-mel）
   本来就比 LHK-Net 弱一档。**结论：量级对得上，"12 类更难"现在有支撑，但还没有到"复现"。**
2. 🔴 **最重要的新事实：模型在诚实协议上的精度超过了"记住船身份"的天花板。**
   `temporal` 的船天花板是 **55.14%**，模型拿到 **61.82%** —— 它**不是**在查船的身份表，
   而是真的学到了声学。这一条比"比文献低 10 个点"重要得多。
3. **泄漏有多大，取决于语料怎么切片段**：同一份代码、同一个模型、同一个配方，
   `clip_random` − 诚实协议在 DeepShip 上是 **30.4 pp**、在 OceanShip 上是 **7.2 pp**。
   机制见上表（`w1_05`）：DeepShip 是首尾相接的分片，OceanShip 是任意时刻的短段。
4. **DeepShip 的 `clip_random` 有 0.92 —— 那是检索，不是识别**：89.6% 的测试片段在训练集里
   有一个**相邻的** 5 s 窗口（同一段音频），模型只要把它匹配出来就够了。
5. OceanShip 仍然弱：`clip_random` 0.3649 / `vessel_disjoint` 0.2926，macro-F1 只有 0.10–0.12，
   只有 Towing / Cargo / Tug 三类有精度，其余整类为 0。**这一点在论文集里必须诚实处理。**

### ✅ W3-A3 v2 已跑（CPU，144.8 s，两个语料）——**"一个标量就够了"**

| corpus | protocol | 元数据天花板 | **1 scalar** | 1-NN 标签 | **1-NN 同录音** | 随机基线 | 模型 |
|---|---|---|---|---|---|---|---|
| DeepShip | `clip_random` | 99.82% | **78.18%** | **83.22%** | **66.78%** | 0.286% | 0.9219 |
| DeepShip | `rec_random` | 0.00% | 34.61% | 50.33% | 0% | 0 | 0.6842 |
| DeepShip | `temporal` | 0.00% | 21.82% | 50.15% | 0% | 0 | 0.6182 |
| OceanShip | `clip_random` | 90.80% | 9.34% | 25.48% | **5.68%** | 0.032% | 0.3649 |
| OceanShip | `rec_random` | 0.00% | — | 22.19% | 0% | 0 | — |
| OceanShip | `vessel_disjoint` | 0.00% | — | 20.13% | 0% | 0 | 0.2926 |

**① 一个标量 = 78%。** DeepShip 的 `src_dur_s` 是**源文件长度**，而构建器把一条文件切成 5 s 窗口
→ 每条录音的所有片段带**同一个数**，时长因此成了录音指纹。**不看音频、不训练**，
只按这一列找最近邻，`clip_random` 就有 **78.18%**（5 seed 78.18–80.02%）；
256 维声学描述子 83.22%；训练好的模型 92.19%。**这个次序本身就是结论：那份 0.92 是检索任务。**
而且它在 `rec_random`/`temporal` 上**塌到 34.61%/21.82%（时长的分类能力≈常数基线）**——
说明 78% 买到的是**身份**，不是船型。

**② 1-NN 同录音率：DeepShip 66.78%（随机 0.286%，233.8×）vs OceanShip 5.68%（180.1×，v1 复现）。**
差了一个数量级，且与 `w1_05` 用纯元数据算出的结构可用率（89.6%）方向一致。

**③ 最近邻在音频流里离多远**：DeepShip `clip_random` 中 **36.8% 恰好是相邻窗口（|Δclip_idx| = 1）**、
55.9% 在 2 以内、79.6% 在 5 以内，中位 **2**。即最近邻基本就是**同一段音频**。
OceanShip 打不出这一栏——`clip_idx` 是构建期的行计数器，**不是流位置**；**"取不到位置"本身就是要报的结果。**

**④ 时长对船型几乎无用**：DeepShip 时长分箱多数标签在 `clip_random` 上 43.93%（常数 27.15%），
但录音级隔离后掉到常数以下。→ 报纸上应把"一个标量 78%"归因于**身份检索**。

### ✅ W3-A 主表已跑（tag `rec2`，33 配置 / **555.3 min = 9.25 h** / 无报错）

| rep | protocol | n | **acc mean ± sd** | min–max | train | predK | F1 | leak%（按录音） |
|---|---|---|---|---|---|---|---|---|
| logmel | `clip_random` | 5 | **0.3672 ± 0.0041** | .3614–.3714 | 0.405 | 7.4 | 0.132 | 85.0 |
| logmel | `rec_random` | 5 | **0.3508 ± 0.0125** | .3364–.3641 | 0.409 | 7.2 | 0.114 | 0.0 |
| logmel | `vessel_disjoint` | 5 | **0.3157 ± 0.0469** | .2770–.3792 | 0.413 | 7.6 | 0.108 | 0.0 |
| spec | `clip_random` | 5 | 0.3647 ± 0.0036 | .3590–.3684 | 0.395 | 7.4 | 0.138 | 85.0 |
| spec | `rec_random` | 5 | 0.3458 ± 0.0129 | .3279–.3569 | 0.402 | 7.0 | 0.112 | 0.0 |
| spec | `vessel_disjoint` | 5 | 0.3120 ± 0.0479 | .2654–.3791 | 0.406 | 6.6 | 0.104 | 0.0 |
| logmel | `logreg`（下限） | 1 | .3208 / .3030 / .2709 | — | 0.455 | 11–12 | .151/.126/.106 | — |

**① 三级阶梯成立，且在两个表征里都成立**（Table 2 / Fig 2 的骨架）。配对同 seed 差值：

| 落差 | logmel | spec |
|---|---|---|
| `clip_random` − `rec_random` | **+1.6 pp**（sd 1.4） | **+1.9 pp**（sd 1.1） |
| `rec_random` − `vessel_disjoint` | +3.5 pp（sd **4.8**） | +3.4 pp（sd **5.1**） |
| `clip_random` − `vessel_disjoint` | **+5.1 pp**（sd **4.5**） | +5.3 pp（sd **4.6**） |

**② 🔴 本文最有力的一句话，从此有模型侧证据**：同一个协议名字，
**OceanShip 上只值 +1.6 pp，DeepShip 上值 +30.4 pp**（0.9219 vs 0.6182）。
而且 **OceanShip 严格协议自己的抽样 sd（4.5 pp）≈ 它要测的落差（5.1 pp）** ——
**单 seed 的严格协议数字不可作为结论**（P11 在模型侧得到确认）。

**③ P3 已答**：`spec` 与 `logmel` 的均值差 ≤0.6 pp、sd 同阶、单调性一致
→ **落差是数据性质，不是特征工程的产物**。

**④ 「模型太弱」这个质疑基本被堵死**：`logreg` 线性探针在每个协议上只比 CNN 低 **2.2–4.4 pp**
（配对 CNN 优势：clip +4.4 / rec +3.9 / vessel +2.2）；rc2 的 11.2M ResNet18 只高 +1.0 pp；
且 ResNet18 把训练集记到 **0.9999** 而测试仍只有 **0.3777** → **天花板不是优化或容量**。

**⑤ 必须写进 limitations 的一件事**：模型实际是**三分类**——Towing 0.749 / Cargo 0.632 / Tug 0.401，
其余 9 类 ≤0.074，其中 **4–5 类从不被预测**（Sailing / SAR / Anti-pollution / Dredging / 偶尔 Tanker）。
而 Towing+Tug+Cargo 占 61.5% 片段 —— **0.367 基本就是这三类的加权。** predK 恒为 7/12，macro-F1 0.11（均匀 0.083）。

### ▶ 下一步（按性价比排序）

**0. 先做免费的：把 W3-A 的输出存成 Kaggle Dataset，然后跑 `w3e_aggregate`（CPU，分钟级、零 GPU）。**
它给出 Table 2/3 的**录音级 bootstrap 95% CI**（1,000 次重采样）与 Fig 2/3 ——
没有这个，Table 2 的误差棒就缺了，而本文批评的正是"只报单点估计"。
（每次 sweep 跑完都重跑一次 W3-E 即可，它自动合并 `/kaggle/input/*` 里的全部 `results.csv`。）

**1. `kaggle_w3c_diagnostics.ipynb`（GPU 2–4 h）** —— Table 7，含三项 ★★★：
P7 **label shuffling**（Hummel 强制，泄漏类论文缺它必被质疑）、P8 **recording-id NMI**、
P5 **时长捷径**（Passenger 中位 3.67 s vs Tug 5.00 s），外加 P6 域探针与 ShipsEar 可分性上界。

**2. `kaggle_w3d_domain_generalization.ipynb`（GPU 4–6 h）** —— Table 5/6 + Fig 5。
**这是"药"**：I4（AIS 域 + IRM/VREx/DANN）、I5（`md5(mmsi)%K` 负对照域）、I7（跨语料零样本）。
**只有批评没有方案的话，AA 会按 incremental 拒。**

**3. `kaggle_w3b_baselines_scale.ipynb`（GPU 5–7 h）** —— Table 4 + Fig 4 的完整 scale 曲线。
价值最低（logreg 与 resnet18 两个点已经说明问题），可放下一周。

**4. 可选（先不做）**：类别加权/平衡采样修三分类坍缩；`vessel_disjoint` 补 10 个 seed（~2.8 h）把严格协议的误差棒收紧。
两条都只在审稿人追问时再做。

**本周预算**：30 GPU h，已用 **9.25**，剩 ~20 h → 够 W3-C + W3-D（6–10 h）再加 W3-B。

### ⚠️ 两个已修的环境坑（本轮踩到）

- **`notebooks/_build_w3_notebooks.py` 曾被 GBK 损坏**：em dash / 箭头 / × / · 共 90 处变成 `鈥?`/`鈫?`/`脳`/`路`。
  已全部修复并重新生成（判别：源文件里**不该出现任何 CJK 或私用区字符**）。
- **生成器的字符串手术会连带删掉相邻函数**：一次 `P_x = r'''` 到 `def nb_w3a3():` 的替换把中间的
  `def nb_w3a4()` 一起切掉了。已从**它生成的那份 notebook** 逆向重建，并用"重新生成后逐字节相同"证明复原正确
  （脚本 `.workbuddy/scratch_20260930/_restore_w3a4.py`）。
  → **规矩：做完手术必须 `ast.parse` + 列出所有 `def nb_*(` 名字自查，别只看 AST 过了没有。**

### 配方是 run 身份的一部分

每个 benchmark notebook 的驱动格顶部有两个常量，**改动是显式的、有后果的**：

```python
RECIPE_TAG = "rec2"                                                  # 进 run_id
RECIPE = ["--epochs", "24", "--norm", "affine", "--lr", "0.003", "--warmup", "3"]
```

改了 `RECIPE` 就要**同时改 `RECIPE_TAG`**，否则重跑会被判成"已完成"而跳过。单个配置可以在
`extra` 里覆盖任意一项（argparse 取最后一次出现的值），W3-A2 就是这么做的。

标签历史：`rec1` / `rc` = 调度未推进的那批，**永远不要复用**。`rec2` / `rc2` = 修复后的。

### 用法（每个都一样）

1. Kaggle → New Notebook → 上传对应的 `.ipynb`
2. Add Input 挂 `under_water_features`（W3-E 再挂前面几个的输出数据集）
3. GPU（W3-E 选 None）→ **Save Version → Save & Run All**
4. 跑完 → **Output 标签页 → Create Dataset**，名字用 notebook 名

**续跑是自动的**：驱动从 `/kaggle/working` 和 `/kaggle/input` 两处读 `results.csv`，按 `run_id` 跳过
已完成的配置。**名字以 `_` 开头的目录是暂存区，永远不参与扫描** —— 探针就写在 `_probe/`。

### 每个 notebook 都带的三道保险

1. **第 1 格先做路径探测**：按 slug 枚举两层目录找 `oceanship/index.csv`，找到就打印三个语料库的
   行数与体积；找不到就打印 `/kaggle/input` 的实际结构，把真实路径填进 `MANUAL` 即可绕过。
2. **跑一次 `--smoke` 接线检查**：会打印解析到的路径、协议、域分组数、泄漏率、`norm` 与 `recipe`。
3. **时间预算守卫**：到点干净退出，重跑就续。

### 旧脚本的位置

`src/w3_00_sweep.py` / `w3_01_run_benchmark.py` / `w3_02_aggregate.py` 是照 **DeepShip-only 的 5.4 GB
`uwatr-logmel`** 写的，**已被下面的三个脚本取代，不要再直接用**：

| 新脚本 | 作用 |
|---|---|
| `src/w3_run.py` | 一个配置 = 一个进程；三语料 × 三协议 × 6 backbone × 4 方法 × 4 域定义；`--norm`；label shuffling；embedding 导出 |
| `src/w3_diag.py` | 冻结表征上的诊断（recording-id NMI、域可分性、时长捷径） |
| `src/w3_agg.py` | 全部表格 + 图，含**录音级 bootstrap CI** 与**泄漏分解**（"见过的船"vs"没见过的船"分开报） |

`src/w3_run.py` 的 `run_id_for()` / `protocol_for()` 是 run 命名的唯一来源，notebook 驱动 import 使用。
notebook 由 `notebooks/_build_w3_notebooks.py` 生成，**嵌入脚本与 `src/` 逐字节一致（构建期断言）**。

---

## 预算核算



| 项目 | 消耗 | 说明 |
|---|---|---|
| W2 特征缓存 | ~0.5 GPU h | 一次性 |
| W3 优先级 1–3 | ~5.5 GPU h | 主结果 |
| W3 优先级 4 | ~7.5 GPU h | 模型规模结论 |
| W3 优先级 5 | ~20 GPU h | 弹性 |
| **主结果合计** | **~6 GPU h** | **不到一周额度的 1/5** |
| 全量合计 | ~33 GPU h | 约 1.1 周额度 |

**结论：主结果一周内能跑完，全量两周内跑完。额度非常宽裕。**

---

## 常见问题

**Q：`/kaggle/working` 只有 20 GB，缓存 5.4 GB 够吗？**
够。缓存本身 5.4 GB，加 predictions（很小）和 results.csv，总占用不到 6 GB。

**Q：为什么用 P100 而不是 T4×2？**
单卡任务 P100 比 T4 快约 45%，而且没有 DDP 通信开销。只有当单卡显存不够时才用 T4×2。

**Q：为什么不用 TPU？**
TPU 对 `soundfile`/`scipy` 这类 CPU 密集预处理极不友好，本项目在 W2 上有大量音频 I/O。

**Q：`logreg` 跑不动 / 内存爆了？**
它在 CPU 上跑，特征展平后是 128×313 = 40,064 维 × 23.6k 样本。如果内存吃紧，加 `--max-train-clips 8000`。

**Q：AST 下载失败？**
需要开 Internet。如果失败脚本会自动退回随机初始化并继续，不会中断 sweep。AST 属于优先级 5，可以先跳过。

**Q：跑出来的精度远低于文献的 99%，是不是我搞错了？**
不是。这正是预期结果。文献的 99% 来自 `clip_random`；严格协议下 DeepShip 的真实水平就是 72% 左右（三个独立团队的一致数字：72.27% / 0.7226 / 72.38%）。**请你自己的实验先复现出这个落差，再谈往上推。** 优先级 1 跑完就该看到这个对比。

---

## 下一步（W5–W6）

W1–W4 产出的是"C2 严谨层"。之后：

- **C1 资源层**：等 ShipsEar 到手后重建其录音级 provenance，公开划分文件
- **C3 科学层**：域偏移归因 —— 用 `deepship_recordings.csv` 里的 `ship_name` / `dist_lo_m` / `dist_hi_m` / `date`，把严格协议下的落差分解到具体因素。**已有素材：84 艘船被记录多次（共 437 条录音），这是现成的同标签跨域配对**
- **C4 方法层**：因子不变域泛化

---

## 本机环境注意事项

- **Bash 工具在本机不可用**（PortableGit 缺核心命令）。一律用 PowerShell。
- **PowerShell 会吞掉 stdout。** 需要看输出时写文件再用 Read 读。
- **`Remove-Item` 会报错但实际删除成功。** 别被错误信息骗到，删完用 `Test-Path` 核验。
- Python 用托管版本：`C:\Users\2101500208gongzihao\.workbuddy\binaries\python\versions\3.13.12\python.exe`
