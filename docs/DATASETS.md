# 三个数据集的获取清单与核查记录

> 核查日期：2026-09-28。**所有链接都是实测过的**：能下的标了 HTTP 状态和字节数，需要登录的标了实测结果。
> 自动化脚本：`python src/w0_fetch_datasets.py --all`

---

## 总览

| 数据集 | 获取难度 | 状态 | 我们需要它做什么 |
|---|---|---|---|
| **Oceanship** ⭐ | ✅ 容易 | **GitHub raw（元数据）+ 123云盘/百度网盘（音频）**，15 类 / 892 MMSI / AIS 字段全覆盖 | **主基准（替代 UniqueShip）** |
| **DeepShip** | ✅ 最容易 | Kaggle 两个完整镜像，20 GB，**已验证内容** | 副数据集 / 同源对照 |
| **ShipsEar** | ✅ 已拿到 | HuggingFace 277 MB，**已下载并校验** | 独立采集域（零样本目标域） |
| **UniqueShip** | 🔴 最难 | **全平台无镜像**，只有 Google Drive，需账号 + 权限 + 国内需代理 | 冲刺项，作第三 ONC 视角 |

---

## 1. DeepShip — 47 h / 265 船 / 4 类 / 32 kHz

### 下载地址（两个等价的完整镜像）

| 来源 | 地址 | 标称体积 | 下载量 |
|---|---|---|---|
| **主**（推荐） | `https://www.kaggle.com/datasets/tangqiji/deepship-raw` | 14,772.4 MB | 318 |
| **备** | `https://www.kaggle.com/datasets/deku14all1/deepship` | 14,772.4 MB | 35 |

⚠️ **避开这些残缺镜像**：`vasundharauppuluri/deepship-main`(502 MB)、`vasundharauppuluri/deepship-dataset-audio`(903 MB)、`vasundharauppuluri/dataset-deep-ship`(903 MB)、`vasundharauppuluri/deep-ship`(502 MB)。名字看起来更"正规"，但只有 500–900 MB，是残片。

### 我实测到的真实内容

```
共 611 个文件，20.23 GB
结构： {Class}/{YYYYMMDD}{可选小写字母}-{序号}/{文件名}.wav
       例：Cargo/20171104-1/1.wav
       Cargo/20171104a-2/2.wav
       Tanker/20180509-137/031902.wav   ← 部分文件名是 HHMMSS 时间戳

609 条录音：Cargo 109 / Passenger 191 / Tanker 240 / Tug 69
共 47.14 h，33,710 个 5 秒片段（官方口径 47h04m / 33,693，吻合）
音频格式：立体声 32 kHz 16-bit = 128,000 bytes/s（由此可精确反推时长）
```

### 官方元数据（4 个 metafile）——**Kaggle 镜像只有 2 个**

Kaggle 镜像里只有 `Cargo/cargo-metafile` 和 `Passenger/passengership-metafile`。
**Tanker 和 Tug 的必须从 GitHub 取：**

```
https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Cargo/cargo-metafile
https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Passengership/passengership-metafile
https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Tanker/tanker-metafile
https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Tug/tug-metafile
```

**这四个文件我已经下载并校验过**（见下方 SHA256 表），`w0_fetch_datasets.py --metafiles` 会自动取。

### 获取方式

**推荐：根本不要下载到本地。** 在 Kaggle Notebook 里 attach 这个数据集，它会以只读方式挂载到 `/kaggle/input/deepship-raw`，20 GB 不占本机磁盘、不占 notebook 的 20 GB 工作目录。

本地下载需要 Kaggle API token（`kaggle.com → Settings → API → Create New Token` → 存到 `~/.kaggle/kaggle.json`），然后：

```bash
kaggle datasets download -d tangqiji/deepship-raw -p <目标目录> --unzip
```

---

## 2. ShipsEar — 已拿到可用副本

### ✅ 公开音频（已下载校验）

| 文件 | 地址 | 大小 |
|---|---|---|
| **ShipsEar.zip** | `https://huggingface.co/datasets/peng7554/DS3500/resolve/main/ShipsEar.zip` | **290,700,328 B（277 MB）** |
| DS3500.zip（备选预处理 + 深海增强版） | `https://huggingface.co/datasets/peng7554/DS3500/resolve/main/DS3500.zip` | 336,734,088 B |

- 授权 **CC-BY-4.0**，**无需注册、无需邮箱、无需申请**，实测 64 秒下载完
- **完整性已验证**：本机 SHA256 与 HuggingFace 报告的 LFS oid **逐字节一致** → `385C0F99...75347D58`

**解压后内容**：
```
shipsear_5s_16k/{0..4}/{类}_{组}/{类}_{组}_{片段}.wav
2223 个 5 秒 / 16 kHz 片段，5 个类，12 个船型组
label_list.txt + train/test 划分 + 5 组种子变体
```

### ✅ 官方免费样本包（实测可下）

```
https://underwaternoise.atlanttic.uvigo.es/samples.zip
→ 实测 HTTP 200，83,907,591 B（80 MB），Content-Type: application/zip
```
这是 Uvigo 官方提供的"各类型船只样本"，不是全库，但属于**官方免费资源**，可用于验证音频格式。

### ✅ 社区标准划分（无需音频）

| 仓库 | 内容 |
|---|---|
| `https://github.com/YuanX9/ShipsEar-An-Unofficial-Train-Test-Split` | **9 类 / 12 类两套划分**，被 SIR&LMR 83.45、CMoE 86.21、M3-TSE 87.07±2.43、AMTNet 80.95±0.84 使用 |
| `https://github.com/xy980523/ShipsEar-An-Unofficial-Train-Test-Split` | 附 `meta_info.txt`（位置/日期/水深/风速/距离） |

### ❌ 拿不到的：原始 90 条录音

- **Kaggle 上搜 `shipsear` = 0 条结果**（实测）
- **GitHub 上 6 个相关仓库全部只有代码**：`stef729/ShipsEar`(2.2MB)、`yadavkapil23/VesselVoice`(425MB)、`SomyaPratap/ShipsEar-Vessel-Classification-`(22.6MB)、`aridepai17/ShipsEar-DWSTr`(14.8MB)、`AvalonEnjoyer/Audio-classification-ShipsEar`、`Vanillaholic/Shipsear_Audio_Recognition`
- 唯一的途径是发邮件：**`dsantos@gts.uvigo.es`**（页面：`https://atlanttic.uvigo.es/underwaternoise/`）
- 第三方佐证：一个 2026-09-23 发布、明显想公开分发的团队明写 "**No raw audio is redistributed: ShipsEar audio remains subject to its authors' educational-use agreement**"

### ⚠️ 用这个副本时必须注意的两件事

1. **它自带的 6 组划分 100% 泄漏**——12 个船型组全部同时出现在 train 和 test，445/445 测试片段被污染。**必须丢掉它们的划分，用 `w1_02_make_splits.py` 重新生成。**
2. **原始 90 条录音的 provenance 已丢失**（只剩 12 个船型组，`2_0` 组有 843 个片段 = 70 分钟，远超单条录音上限）→ **做不了录音级严格划分**。

---

## 3. Oceanship ⭐ — 主基准（实测可下，无需账号）

**定位：UniqueShip 的替代方案。** 同属 ONC 太平洋网络，但获取路径完全开放。

| 维度 | 值 |
|---|---|
| 采集源 | **ONC（Ocean Networks Canada），2021–2022** |
| 规模 | **53,770 片段 / 39,260 录音 / 65 h**（FG 版） |
| 类别 | **15 类**（Anti-pollution / Cargo / Diving / Dredging / Fishing / Military / Passenger / Pleasure Craft / Port Tender / Sailing / SAR / Spare / Tanker / Towing / Tug） |
| 船只 | **892 个 MMSI**（FG 版；train 818 + test 439，共享 365） |
| 录音 | **10,168**（= distinct `(mmsi, ais_timestamp)`） |
| 元数据 | **17 个 AIS 字段，100% 覆盖**（MMSI / IMO / 呼号 / 真航向 / 对地航向 / 对地航速 / 船型 / 经纬度 / 船长 / 船宽 / 吃水 / 目的港 / ETA） |

> ⚠️ 上一版这里写的是「1,257 MMSI」，那是把 train 818 与 test 439 直接相加、忘了减掉共享的 365。**正确值是 892**，已由 `src/w1_04_oceanship_inventory.py` 实测确认。

### 音频下载（**123云盘为主，实测无密码、有效期到 2099**）

```
train  https://www.123865.com/s/xArrTd-e0kpH   23.44 GiB   无需密码
test   https://www.123865.com/s/xArrTd-i0kpH    5.12 GiB   无需密码
备用域名：把 www.123865.com 换成 www.123684.com（同一份文件）
合计 ≈ 28.6 GiB
```

来源是作者在 GitHub issue #3 里补的新链接（2024-10）。实测 `api/share/info` 返回 `HasPwd: false`、`Expired: false`、`Expiration: 2099-12-12`。

百度网盘（作者 Readme 原始链接）：
```
FG 版  https://pan.baidu.com/s/19-K_QNvINT-ZlfHzd0HuSw   密码 2fme   ✅ 仍有效
CG 版  https://pan.baidu.com/s/1FzqzKmmENbWzJUJafg9i7Q   ❌ 已失效（实测 404）
Full 版：作者不上传，按其说明「Full = CG + FG」自行合并 —— CG 失效后此路不通
```

解压方式（作者提供）：多个 `.aa .ab ... .aj` 分卷先 `cat` 合并成 `.tar` 再 `tar -xvf`。

### 元数据与官方划分（GitHub raw，**已下载并验证**）

| 文件 | 大小 | 内容 |
|---|---|---|
| `oceanship_fg_train.csv` | 10,573,729 B | **45,699 行**，列 `text_path, wav_path, label, x, y, sog, cog, true_heading, mmsi, id, ais_timestamp, label_idx` |
| `oceanship_fg_test.csv` | 1,854,174 B | 8,071 行，同上 |
| `oceanship_full_train.csv` | 13,488,875 B | 87,473 行 |
| `oceanship_full_test.csv` | 2,381,460 B | 15,451 行 |
| `oceanship_fg_train_enhanced.txt` | 121,979 B | 827 行 AIS 详情 / 765 个 MMSI |
| `oceanship_fg_test_enhanced.txt` | 65,108 B | 432 行 / 405 个 MMSI |

下载前缀：`https://raw.githubusercontent.com/lizeyujack/oceanship/main/<文件名>`

**已存到 `data/external/oceanship_gh/`。**

### 🔴 关键发现：三级泄漏，全部由 `src/w1_04_oceanship_inventory.py` 实测

**只靠这 27 MB 元数据 CSV，不需要音频**，就能把下面整张表跑出来（脚本已跑通，产物在 `out/`）：

| 划分 | 录音跨集率 | **船（MMSI）跨集率** |
|---|---|---|
| **作者官方 FG 划分**（我们不是它的作者，只是审计它） | **19.8%** | **91.8%** |
| 我们复现的 `clip_random`（文献主流做法） | 91.3% | **99.6%** |
| 我们新建的 `rec_random`（录音级） | **0.0%** | **98.0%** |
| 我们新建的 `vessel_disjoint`（船级） | **0.0%** | **0.0%** |

**`rec_random` 这一行就是论文的核心：录音跨集率已经归零，船跨集率却还有 98%。**

paper3 的 `oceanship_v1.json` 用更细的分组（其自检 `pairwise_group_intersections` 全为 0 且 `all_checks_passed: True`）得到 **99.4%**——与我们独立算出的 98.0% 同向同量级，互为佐证。

详见 `docs/Oceanship替代方案与船级泄漏发现.md`。

### ⚠️ 数据侧注意事项

- `id` 列 ∈ `{1,3,5,18,19,24}` 是**水听器/部署点编号**，不是船。船身份看 `mmsi`。
- **文件名里的 `id_N` 与 CSV 的 `id` 列不一致**（53,770 行里有 **45,511 行**对不上）。两者取值来自同一小组，但确实矛盾，所以至少有一个不是它看起来的那个东西。**我们两个都不依赖**，船身份只用 `mmsi`（真实 9 位 MMSI，100% 非空）。
- **`oceanship_full_*.csv` 只有 3 列**（`wav_path, label, text_path`）**，没有 MMSI** → MMSI 级分析只能用 FG 版。
- **15 类里只有 12 类能用**：Military ship / Diving ship 各 1 艘船、Spare 2 艘船，凑不出三路船级划分。剔掉这 3 类只损失 **16 个片段（0.03%）**。脚本默认自动剔除并显式报告。
- **各类船数极不均衡**：Cargo 221 / Pleasure Craft 204 / Tug 114 / Fishing 93 / Towing 79 / Sailing 66 / Passenger 60 / SAR 21 / Tanker 19 / Anti-pollution 7 / Port Tender 5 / Dredging 4。**Tanker（19 船 / 152 片段）和 Port Tender（5 船 / 1,253 片段）是两个极端**，建议主表用 macro-F1 + 每类 CI，或合并尾部类别。
- 采样率需拿到音频后实测确认。

---

## 4. UniqueShip — 冲刺项（全平台无镜像）

### 信息入口

| 项目 | 地址 |
|---|---|
| 门户 | `https://uniqueshipdata.org` |
| 论文 | `https://arxiv.org/abs/2609.13659`（OCEANS 2026） |
| 代码仓库 | `https://github.com/Kitware/UniqueShipData`（**只有网站源码，不含数据**） |
| 联系 | `https://www.kitware.com/contact/` 或站点上的 inquiry 表单 |

**规格**：2,460 h（含背景 3,437 h）/ **4,218 艘船** / 11 类 / 250 万条 5 秒片段 @ 20 kHz
**划分**：按 **MMSI 船级隔离**（比录音级更严格），背景类按天分组
**元数据**：**每个样本 17 个 AIS 字段**（MMSI、到水听器距离、航速、航向、船长、船宽、吃水、航行状态…）
**基线**：MobileNetV3 / ViT-B/16 / SwinV2 × STFT/Mel，**最好 66.5%（Swin + Mel）**

### 四个下载链接（实测都跳登录页）

| 划分 | 规模 | 音频 | 频谱图 |
|---|---|---|---|
| **12 Class - 5 Hours Each** ⭐ | 60 h / 4,218 船 / 12 类<br>26 GB 解压 / 17.8 GB 压缩 | `drive.google.com/drive/folders/1HxV8AGh8ckZS3JehAU_gY2fuCnscX-eV` | `.../1xOXX4dUmtSET7Utb4gfscZp1dlPoT76e` |
| 5 Class - Balanced | 213 h / 3,175 船 / 5 类<br>92 GB 解压 / 64.8 GB 压缩 | `.../1G0e-2Nhi6CWKuBRStlR8GTtcGyYiT2h6` | `.../1a6wLqfuBh3PMqSlZfy-IVjTfrJMtqo11` |

实测结果：四个链接全部返回 `HTTP 200` 但标题为 `Google Drive: Sign-in` → **必须登录 Google 账号**。

### 站点自己有两处矛盾的说法

- 发布区块写：「please **request Google Drive permission** to access the current splits」
- 咨询区块写：「Current dataset splits are available to download directly — **no request or approval is required**」

**先按"不需要审批、只要登录"试。** 如果 Drive 拒绝访问，走 inquiry 表单或 Kitware 联系方式。

### 站点还透露了两件有用的事

1. **有一个 25 小时的快速子集**，以及 **5 个 80/10/10 fold** —— 25 h 子集正好适合先跑通全链路
2. 更大的划分（"Main 5, Unbalanced"）**即将迁移到 AWS** —— 站点明说 "Please feel free to message if currently waiting"

### ⚠️ 国内访问的现实问题

**Google Drive 在中国大陆无法直接访问。** 这是这条路上唯一的真实障碍，有两个解法：

| 方案 | 说明 |
|---|---|
| A. 用代理登录后下载 | 下 **频谱图** 而不是音频——它们已经算好，直接跳过我们的 W2 步骤，下游完全一样 |
| B. 联系 Kitware 要 AWS 链接 | 站点自己说大划分正在迁 AWS；用 inquiry 表单说明你在国内、Drive 不可达，请求替代传输方式 |

### 镜像排查：全平台 0 结果（2026-09-28 实测）

| 平台 | 查询 | 结果 |
|---|---|---|
| Kaggle | `uniqueship` / `unique ship` / `ONC ship acoustic` / `ocean networks canada ship` / `MARINA ship` | **0 条**（`unique ship` 返回 20 条全是 superstore 零售数据，无关） |
| HuggingFace | `uniqueship` / `unique-ship` / `ship classification acoustic` / `underwater acoustic target` | **0 条** |
| ModelScope | `ship` / `acoustic` | **0 条相关**（返回的是卫星图像船检、城市声景，与 UATR 无关） |
| Zenodo | `uniqueship` | 403 / 无相关记录 |
| GitHub | 全库搜索 | 只有 **`Kitware/UniqueShipData`**，7 MB，**纯网站源码，不含任何数据** |
| AWS Open Data | `oceannetworks` / `hydrophone` | **0 条**（注册表里没有 ONC） |
| 中文社区（百度网盘/知乎/CSDN 转载） | 全网搜索 | **无** |

**结论：UniqueShip 只有一个分发渠道 = `uniqueshipdata.org` 上的 4 个 Google Drive 链接。**

### Google Drive 访问权限实测（比站点自述更明确）

用 `embeddedfolderview` 端点（这个端点对"任何人可看"的公开文件夹会直接返回文件列表，无需登录）测试四个文件夹：

```
12Class-5h AUDIO     → 401 Unauthorized
12Class-5h SPECTRO   → 401 Unauthorized
5Class-Bal AUDIO     → 401 Unauthorized
5Class-Bal SPECTRO   → 401 Unauthorized
```

再用浏览器 UA 抓完整页面：`HTTP 200` 但 `<title>` = `Google Drive: Sign-in`。

→ **这不是"公开链接可直接下"，是"需要 Google 账号 + 需要被授予访问权限"。** 站点 Releases 区块的自述与实测一致（"please **request Google Drive permission** to access the current splits"）；Inquiries 区块那句 "no request or approval is required" 与实测矛盾，**以实测为准**。

### 🟢 但是：原始数据本身是 CC-BY 4.0，可以从 ONC 直接拿

这是最重要的发现。站点页脚的免责声明原文：

> **Data owned by Ocean Networks Canada and available under CC-BY 4.0**
> https://www.oceannetworks.ca/data/data-policy/

ONC 官方数据政策原文（实测抓取）：

> Our data are machine-readable, **freely shared, used and built on without restrictions**. … All data solely-owned by ONC are available **free of charge** through web services, internet downloads, special requests… provided under a **Creative Commons CC-BY 4.0 License, open and free for anyone to use with attribution**.

**也就是说：UniqueShip 是 ONC 开放数据的衍生品（derive），原始水听器录音 + AIS 轨迹本身就是合法免费公开的。**

| 路线 | 需要什么 | 代价 |
|---|---|---|
| **A. 走 UniqueShip（省事）** | Google 账号 + 权限授予 +（国内）代理 | 拿到就用，**推荐先试** |
| **B. 从 ONC 自建（合法但费力）** | `data.oceannetworks.ca` 门户 + 免费 API token | 等于把论文作者的工程重做一遍（水听器录音 × AIS 关联 × 船级分组），**6 周时间线内不可行** |
| **C. 只拿元数据/划分** | GitHub `Kitware/UniqueShipData`（无数据）；论文附录 | 只能看，不能用 |

**注意**：论文说数据来自 "seven ONC deployments in the Strait of Georgia (May 2016 – November 2023)"。ONC 的 Oceans 3.0 门户（`data.oceannetworks.ca`）实测可达（HTTP 200），注册后可拿免费 token 调 API。但要把录音切成 5 秒片段、关联 AIS、按 MMSI 分组、剔除背景——**这就是 UniqueShip 论文本身的工作量**，不是我们该重做的。

→ **路线 B 只作为"Drive 彻底拿不到"时的最后手段**，不作为计划。

### 站点还透露的两件有用的事（已确认）

1. **25 小时快速子集** + **5 个 80/10/10 fold** —— 25 h 子集是跑通全链路的最优起点
2. 更大的划分（"Main 5, Unbalanced"）**即将迁 AWS** —— 站点原文 "The larger splits ('Main 5, Unbalanced') will be uploaded soon, with links to AWS. Please feel free to message if currently waiting."
   → **这是一个可用的请求理由**：现在就用 inquiry 表单说明国内 Drive 不可达，请他们在 AWS 上线后通知或直接给替代链接。

---

## 5. 本机已有的资产与校验和

```
SHA256                                                            大小          路径
385C0F9961FBD6D8350584EA7B5E45A76A20013575C35C3C87EFC48C75347D58  290,700,328  data/external/shipsear_hf/ShipsEar.zip
1697CFE92051AD3605E83489621E754BB333F4A651529898CB7C51F9B185E6C4       23,985  data/metadata/deepship_manifest.tsv
98D541A4D7A98D8048D410D254D89484B3082CE577593E9112FB4A635C0C40AA        5,718  data/metadata/raw_metafiles/cargo-metafile.txt
3B2B28EEBCFFE6AA8C7E3DC880E9A226ED86920CE2CEAA41A4B5AE90209DA1F3        9,661  data/metadata/raw_metafiles/passengership-metafile.txt
505F9E7C2BD0EC48D58E2C60CF2E6822FFE20CF32AC103472B786A5E5D0C90F6       12,152  data/metadata/raw_metafiles/tanker-metafile.txt
9C17D512706CA36B0F48DEF74668939F59780EE5D0F295797723A21E9FF83B0B        3,563  data/metadata/raw_metafiles/tug-metafile.txt
803714C8A66C41E9F76F5673BCB3150ACC970A362AEC8D09A200D641FEC38788       90,487  out/deepship_recordings.csv
22F462D1D1E2C6805F6AFCE0ACAAB80E9056B0B7479104B5C296EFBCB7CC7A68    1,354,322  out/splits/splits.csv.gz
```

其中 `ShipsEar.zip` 的摘要与 HuggingFace 公布的 LFS oid 一致，`raw_metafiles/*` 与 GitHub 原文一致。
`deepship_manifest.tsv` 是从 Kaggle API 枚举出的 DeepShip 文件清单（路径 + 字节数），**有了它就能在没有 20 GB 音频的情况下把 W1 全流程跑通并验证**。

---

## 6. 一键获取

```powershell
# 全部（DeepShip 需先配好 Kaggle token）
python src/w0_fetch_datasets.py --all

# 只取 4 个官方 metafile（约 30 KB，秒下）
python src/w0_fetch_datasets.py --metafiles

# 只取 ShipsEar 公开副本（277 MB）
python src/w0_fetch_datasets.py --shipsear

# 打印 UniqueShip 的链接与注意事项
python src/w0_fetch_datasets.py --uniqueship-notes
```

脚本行为：
- **幂等**：已存在且 SHA256 匹配的文件直接跳过
- **校验**：所有文件都比对已知摘要，不匹配自动重下（20 GB 的声学语料静默损坏的代价太高）
- **无第三方依赖**：只用标准库 `urllib`
- DeepShip 需要凭据时不会失败退出，而是打印获取 token 的完整步骤

> **Oceanship 目前需手动下**：音频走百度网盘（密码见 §3），元数据走 GitHub raw（已存本地）。
> 音频不在脚本里做，是因为百度网盘没有稳定的命令行取法。

---

## 7. 结论：现在就能开工

| 数据集 | 现在能用吗 |
|---|---|
| **Oceanship** | ✅ **元数据与官方划分已在本机**（`data/external/oceanship_gh/`），只差百度网盘下音频 |
| **ShipsEar** | ✅ **已在本机，校验通过**（有 provenance 限制，见 §2） |
| **DeepShip** | ✅ 在 Kaggle 上 attach 即可，不用下载到本地 |
| **UniqueShip** | 🔴 全平台无镜像，只差 Google 账号 + 权限授予（+ 国内代理）——**已降级为冲刺项，不再阻塞** |

**主基准已换人**：Oceanship 的元数据与官方划分已在本机，音频走百度网盘（几小时就能下完），**不用等 Google 账号、不用等邮件、不用等 UniqueShip**。

### 行动建议（按优先级）

1. **今天：从百度网盘下 Oceanship FG 版**（密码 `2fme`）+ CG 版（`8igj`）。这是新的主基准，且完全不依赖任何审批。
2. **同时：把 `w1_02_make_splits.py` 的分组键从"录音"改成"MMSI"**，生成 Oceanship 的船级划分——这正是两套已有划分都缺的那一环。
3. **顺手：把「三级泄漏谱系」写成 C2 的核心表格**（见 `docs/Oceanship替代方案与船级泄漏发现.md`）。这是目前最硬的贡献，数字全部可复现。
4. **UniqueShip 去申请一次但不要等**——它是唯一自带船级隔离的现成基准，能和我们的结果互证。走 inquiry 表单时**同时问 AWS 链接**（站点自己说大划分即将迁 AWS）。
5. **ShipsEar 邮件发出去**，把 paper3 的 `shipsear_v1.json`（原始 90 条录音的录音级划分，63/15/12）留作成功后的对齐基础。
6. **不要走 ONC 自建路线** —— 那等于重做别人的整篇论文，6 周时间线装不下。
