# Flame in Freefall 后端

输入材料、氧浓度、压力和对向气流，返回持续燃烧时的蔓延速度、名义 90% 速度预测区间、不持续燃烧概率、可燃性地图、SHAP 解释及下一次实验建议。

**当前v0.3模型训练于2000条合成数据＋11条逐行核验的真实NASA观测，`data_source="mixed"`。** NASA训练数据仅覆盖一种薄纤维素、一个名义压力和5.18秒观测窗；未在窗内熄灭不代表无限期持续燃烧。物理参数仍是黑客松近似，尚不能用于航天器火灾安全判断。前端应保留来源与适用范围；速度是条件速度，熄灭时仍可能显示正值。

可查询的NASA证据共183条：原始数值表13条、PSI/BASS-II官方目录129条、BASS-II报告的39个速度阶段＋2个淬熄边界。只有11条具有完整可用目标进入训练，2条左删失速率保留为证据，其余目录/不完整工况未被补造成训练标签。数据、来源、许可及映射见 [NASA_DATA.md](docs/NASA_DATA.md)。

v0.2 新增一套可演示的研究决策能力，保持所有原有接口的字段类型不变：

| 新接口 | 能直接展示的能力 |
|---|---|
| `POST /predict/batch` | 一次向量化预测最多256个条件，结果与逐个调用一致 |
| `POST /analyze` | 比较5材料、4大气预设、3个控制参数敏感度，传播输入容差并标记外推 |
| `POST /optimize` | 搜索三维约束网格，验证中心和8角点，返回最小改动方案与Pareto前沿 |
| `POST /experiments/design` | 展示集成信息增益代理、成员内熵、分歧、分数分解、多样性和已测点排除 |
| `GET /model-card` | 查看折外可靠性分箱、ECE、材料/实验切片、4类SHA256指纹与模型身份 |
| `GET /datasets` 及其观测子接口 | 查询真实NASA源文件、逐条记录、排除原因，以及是否实际用于当前模型 |

前端接入见 [高级 API 契约](docs/ADVANCED_API.md)；两分钟演示顺序见 [演示说明](docs/DEMO.md)。接口返回请求追踪ID、耗时与模型ID响应头，方便队友定位联调问题。

## 前端先看这里

- 接口契约：[docs/API.md](docs/API.md)，包含完整 JSON 示例、单位、错误格式及 TypeScript 类型。
- 开发服务：[http://localhost:8000](http://localhost:8000)，Swagger：[http://localhost:8000/docs](http://localhost:8000/docs)，OpenAPI：[http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)。
- 已允许 `http://localhost:3000`、`http://localhost:5173` 的 CORS。
- `POST /predict` 按队友的 TypeScript 返回数字 `spread_rate_mm_s`、`spread_ci`、`extinction_prob`、`extinction_ci`、`in_data_range`。也接受 `o2` 和材料别名 `cotton`。
- `POST /predict/detailed` 支持最初的对象格式：`spread_rate_mm_s: {mean, lower, upper}`。两个入口格式固定。
- 地图和解释同时支持 GET、POST；推荐结果包装为 `{points: [...]}`。
- 地图矩阵 **行是 flow，列是 O2**：`z[i][j]` 对应 `y[i]`、`x[j]`。`z` 是熄灭概率。

## 从零运行（PowerShell）

需要 Python 3.11+，本工作区使用 Python 3.12.14。无须激活虚拟环境：

```powershell
Set-Location E:\Hackathon\backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/generate_data.py
.\.venv\Scripts\python.exe scripts/import_nasa.py
.\.venv\Scripts\python.exe scripts/train_model.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

现有 `.venv` 可直接复用。macOS/Linux 使用 `python3 -m venv .venv`、`.venv/bin/python` 执行相同命令。

生成器仅写 `data/experiments.csv`（2000条合成、20组、seed42）；导入器另写 `data/experiments_mixed.csv`（2011条、21组），不覆盖合成源文件。训练默认使用mixed文件、各20个回归/分类器和5折分组验证，写入 `artifacts/model.joblib` 与 `artifacts/metrics.json`。生成CSV和模型不入Git，核验过的NASA原始值快照与SHA清单会入Git，支持离线复现。

需要重新下载并校验全部5个官方源文件时运行：

```powershell
.\.venv\Scripts\python.exe scripts/import_nasa.py --download-verify
```

源文件保存在 `artifacts/nasa_sources`，包括约49MB的BASS-II报告。PSI文件通过其官方网站的公开下载接口获取，不保存临时签名URL。

服务仅在启动时加载模型；重新训练后应重启服务。缺少或损坏模型时 `/health` 仍返回 200、`status="model_not_loaded"`，推理返回带操作说明的 503，不会在请求中自动训练。

可直接验证前端请求：

```powershell
$predictionBody = @{material="cotton"; o2=21; pressure_kpa=101.3; flow_cm_s=10} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType application/json -Body $predictionBody
Invoke-RestMethod 'http://localhost:8000/map?material=cotton&pressure_kpa=70'
Invoke-RestMethod 'http://localhost:8000/suggest?material=cotton&k=5'
Invoke-RestMethod 'http://localhost:8000/datasets'
```

运行服务后，在另一个终端测量完整 HTTP 地图延迟：

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_map.py
```

本机正式20成员v0.3模型实测完整HTTP：50×50地图冷请求中位数 **0.221秒**、最大 **0.237秒**；缓存请求中位数 **0.025秒**。场景分析约 **0.051秒**，默认三维约束搜索约 **0.499秒**，信息实验设计约 **0.147秒**。这是本机测量值，不是任意硬件的延迟保证。服务按模型实例缓存最近32个地图；换模型会重建缓存。

完整串联新能力并保存可复核的JSON证据：

```powershell
.\.venv\Scripts\python.exe scripts/demo.py
```

输出保存在 `artifacts/demo_report.json`（不入Git），包含每次请求、完整响应、请求ID和实测耗时。最大优化范围允许4096个网格中心加当前条件，并检查其8角点；最重档位会比默认搜索慢，工作量由输入schema限制。

## 模块如何配合

| 模块 | 可以向评委解释的职责 |
|---|---|
| `physics/flame_spread.py` | 热平衡速度 + U 形氧极限，显式区分低流淬熄、高流吹熄和不能点燃 |
| `data/materials.py`、`presets.py` | 近似材料参数和带出处/示意标签的大气情景 |
| `data/loader.py` | 严格校验同一 CSV 格式、单位范围和燃烧标签一致性 |
| `data/synthetic.py` | 加入平滑隐藏偏差、实验批次偏移和噪声，并留下参数空洞 |
| `data/nasa.py`、`nasa_curated/` | 核验NASA源文件及快照哈希，保留原始值/删失记录，严格转换11条完整观测 |
| `models/base.py` | 可替换模型协议；HTTP 层只读取统一预测结果 |
| `models/residual_ensemble.py` | 整批 bootstrap 的 LightGBM 残差集成、熄灭概率及局部数据支持 |
| `models/train.py` | 三种模型共用 GroupKFold，保存实验编号审计和折外指标 |
| `services/flammability_map.py` | 批量预测网格、提取 P=0.5 边界、缓存地图 |
| `services/suggest.py` | 不确定性 × 边界偏好 × 数据稀疏程度，保留候选多样性 |
| `services/explain.py` | 对全部成员做 SHAP，返回全局重要性及局部加和贡献 |
| `services/decision.py` | 单批场景分析、可复现输入扰动、受约束的邻域检查与Pareto搜索 |
| `services/experiment_design.py` | 用全成员熵差设计下一批实验，并保留分数依据 |
| `models/provenance.py`、`validation.py` | 训练身份与运行时一致性验证，折外校准及切片报告 |
| `services/model_card.py` | 为前端提供可审阅的模型能力、数据覆盖和限制说明 |
| `app/schemas.py`、`api/routes.py`、`main.py` | Pydantic 校验、固定接口、CORS 与启动加载 |

## 三种模型的实际结果

以下来自当前 `artifacts/metrics.json`：2000条synthetic＋11条NASA，共21个分组，seed42，GroupKFold5折。NASA整篇报告作为同一保守实验组，全部11条件同折留出。每行恰好一次折外预测，训练/测试组无交集；测试集不参与拟合、噪声或支持域估计。与旧synthetic-only结果的折组成不同，不能把聚合分数差异直接归因于加入NASA。

| 模型 | log MAE ↓ | log RMSE ↓ | 熄灭 ROC-AUC ↑ | Brier ↓ | ECE ↓ | 90% 速度区间覆盖率 |
|---|---:|---:|---:|---:|---:|---:|
| 纯物理 | 0.2720 | 0.3381 | **0.9849** | 0.0484 | 0.0167 | 无区间 |
| 纯机器学习 | 0.2027 | 0.2955 | 0.9754 | 0.0599 | 0.0242 | 94.70% |
| 物理 + ML 残差 | **0.1274** | **0.1623** | 0.9847 | **0.0476** | **0.0127** | 93.48% |

必须同时查看NASA整组留出切片（只有8条可测速率＋3条观测熄灭、1个报告组）：

| 模型 | NASA log MAE | NASA log RMSE | NASA ROC-AUC | NASA Brier | NASA速度区间覆盖率 |
|---|---:|---:|---:|---:|---:|
| 纯物理 | **0.3542** | **0.4437** | 0.8750 | **0.2163** | 无区间 |
| 纯机器学习 | 0.4451 | 0.5645 | 0.6667 | 0.4555 | 12.5% |
| 物理 + ML 残差 | 0.4074 | 0.4945 | 0.8750 | 0.3459 | 37.5% |

这张真实数据表说明域迁移尚不足：聚合93.48%覆盖率不能代表NASA切片；此切片上的纯物理速度还优于ML模型。数据量和独立campaign不足，不能声称NASA泛化已通过。原synthetic-only模型对该报告的独立检验保存在 `artifacts/synthetic_baseline/nasa_external_validation.json`（log MAE0.410、覆盖率37.5%）。需要从新checkout复现基准时：

```powershell
.\.venv\Scripts\python.exe scripts/train_model.py --data data/experiments.csv --artifacts artifacts/synthetic_baseline
.\.venv\Scripts\python.exe scripts/validate_nasa.py
```

`validate_nasa.py`会拒绝把已经见过该NASA报告的模型当成外部验证对象。

速度误差和覆盖率只计算标注燃烧且有正测速率的行；NASA的燃烧状态仅对应5.18秒观测窗。为比较同一条件速度，纯物理使用熄灭门控前的正潜在速度。纯物理概率由固定氧边界sigmoid构造，并非已校准概率。纯ML使用7个原始/材料特征，不含物理特征；混合模型增加3个物理特征并学习 `ln(V_observed)-ln(V_potential)`。

模型间分歧与观测噪声分开处理：速度区间合并集成对数方差和训练组 OOB 残差噪声，名义水平为 90%，覆盖率由折外数据实测。`extinction_ci` 只反映成员概率的离散；外推时即使成员一致，也不能视为可信，所以另外返回 `in_data_range` 和启发式 `confidence`。

SHAP 的加和目标是平均成员对数残差或平均成员 log-odds；经过 exp/sigmoid 后的平均输出不同。解释对象及单位已写在 API 文档和响应中。

新增可靠性报告使用同一批折外预测，**没有用外层测试标签重新拟合或调参**。ECE基于10个等宽概率箱，必须连同样本数阅读；Brier同时反映校准与区分能力，报告显式保留分箱分解残差。新产物记录数据、物理/材料、训练配置和拟合模型的指纹，以及Python/依赖版本。模型/metrics不同身份不会混用；当前运行的物理源码、材料或推理常量改变时，新指纹模型拒绝加载，要求重训。旧产物保持兼容，但模型卡明确标记验证证据不可用。

## 接入更多真实 NASA 数据

先将合法取得的原始观测整理为以下**完全相同**的 CSV 表头，每行一次测试：

```csv
experiment_id,campaign,material,thickness_mm,o2_percent,pressure_kpa,flow_cm_s,gravity_g,ignited,extinguished,spread_rate_mm_s,source
BASS-batch-001,BASS,pmma_thin,0.25,21,101.3,10,0,true,false,0.6,nasa
```

上面一行是格式示例，**不是 NASA 测量数据**。所有单位必须先手工统一：O2 是百分数（21），压力 kPa，气流 cm/s，厚度 mm，速度 mm/s，重力为地球 g 的倍数。范围检查能捕捉常见错误，但无法识别所有单位误用。

`experiment_id` 应表示真实相关批次，并跨 campaign 唯一，如 `BASS-batch-001`，不能给每行单独编号来伪造独立实验。布尔列接受 true/false 或 1/0；持续燃烧需要正速度，未点燃或已熄灭的速度留空或填 0；`extinguished=true` 必须同时 `ignited=true`。`source` 只允许 `nasa` 或 `synthetic`，模型自动汇总成 `nasa`、`synthetic` 或 `mixed`。

当前模型只覆盖微重力对向流场景：训练拒绝 `gravity_g > 0.01`。不同实验几何、点火方式、并向流与对向流不能在没有科学论证时直接合并。新材料需先在 `data/materials.py` 加入经核验的属性和稳定 ID；真实材料厚度应填在每行中。

已有真实NASA数据通过 `scripts/import_nasa.py`接入。对于未来新增的完整观测，另存文件并显式指定，不要覆盖合成源文件：

```powershell
.\.venv\Scripts\python.exe scripts/train_model.py --data data/nasa_experiments.csv
```

现有混合研究模型可直接重新训练：

```powershell
.\.venv\Scripts\python.exe scripts/train_model.py --data data/experiments_mixed.csv
```

至少需要5个可分离实验组（默认5折），同时包含燃烧和不持续燃烧观测，训练折内也需足够燃烧组。当前NASA报告只能算1保守组，因此不单独训练并宣称独立验证的NASA-only模型。重新训练后重启服务并检查来源/行数；地图NASA测试点同时标明campaign和真实厚度。`generate_data.py`默认只写synthetic源CSV，`import_nasa.py`另写mixedCSV并防止重复导入。

环境变量 `FLAME_DATA_PATH` 可改变默认 CSV，`FLAME_ARTIFACT_DIR` 可改变模型/指标目录；路径配置在 `app/config.py`，不依赖运行命令的工作目录。

## 插入后续 PINN

实现 `models/base.py` 的 `Predictor`：`fit`、`predict`、`save/load`、`support_distance`，提供训练数据和来源/版本元数据；`predict` 返回同一 `PredictionBatch`。使用 `create_app(predictor=your_predictor, metrics=your_metrics)` 即可复用预测、地图和推荐的 HTTP 格式。默认启动加载器位于 `app/main.py`，可替换为该模型的 `load`。

当前 SHAP 服务针对树集成；PINN 需提供对应解释适配器，未提供时 `/explain` 清晰返回 501。项目当前未引入 PyTorch。

普通 `Predictor` 可继续复用批量预测、场景分析、约束搜索、地图和原推荐。信息实验设计需要额外实现 `MemberPredictor.predict_members`；缺少时该接口返回501。外部模型可以不保留原始训练行，模型卡将返回训练摘要不可用，绝不从训练预测伪造折外评估。新指纹验证规则仅解释本项目的ResidualEnsemble产物；外部模型负责自己的身份验证。

## 假设、来源与验证

全部物理/材料/生成器近似常数、模型超参数及展示阈值见 [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md)。这些值集中在各模块的常量表中，便于答辩与替换。**最终 pitch 前请对照 NASA 资料核实大气预设与材料参数。**

资料依据是微重力熄灭存在低流淬熄和高流吹熄的机制，不是对本项目数值公式的标定：[NASA U 形边界机制](https://ntrs.nasa.gov/citations/20230000860)、[NASA 对向流熄灭准则](https://ntrs.nasa.gov/citations/20040053557)。探索大气的 34% O2 / 8.2 psia 背景见 [NASA 探索大气研究](https://ntrs.nasa.gov/citations/20230015380)。验证和解释方法参见 [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)、[TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html)。

当前 **251项pytest全部通过**：在原206项基础上新增NASA源值/单位转换、删失排除、SHA防篡改、3个数据接口、按source评估、全报告整组留出、真实mixed模型/地图来源传播、证据快照身份，以及改名/改source也不能把训练数据冒充外部验证等测试。测试独立生成临时产物；普通测试不依赖网络或当前正式模型。当前环境另有一条Starlette/httpx弃用提示，不影响结果。
