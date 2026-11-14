# Flame in Freefall

基于微重力燃烧数据的 AI 火灾安全洞察黑客松项目。后端提供带不确定性的燃烧预测、可燃性地图、SHAP 解释和主动学习实验推荐。

v0.2新增批量预测、完整场景分析、约束内的稳健研究搜索、信息驱动的实验设计，以及带折外可靠性诊断和数据指纹的模型卡。原有前端接口保持兼容；新增能力见 [高级API契约](backend/docs/ADVANCED_API.md)，演示流程见 [两分钟演示](backend/docs/DEMO.md)。

v0.3已接入NASA官方真实数据：183条可追溯证据可查询，其中11条完整薄片观测加入2000条synthetic形成mixed训练集。NASA仅覆盖一个材料/压力和5.18秒观测窗，真实迁移评估与聚合指标分开报告；近似物理与现有模型尚不能用于火灾安全判断。来源与接入流程见 [NASA数据说明](backend/docs/NASA_DATA.md)。

前后端共同维护接口约定：[backend/docs/API.md](backend/docs/API.md)。前端优先使用 `POST /predict` 的数字速度格式；`POST /predict/detailed` 提供最初的对象格式。

| 路径 | 内容 |
|---|---|
| `backend/` | Python / FastAPI 后端、训练脚本、测试和接口文档 |
| `frontend/` | 后续由前端队友接入的位置 |

完整安装、生成数据、训练、测试和启动说明见 [后端 README](backend/README.md)。全部近似常数见 [假设清单](backend/docs/ASSUMPTIONS.md)。

后端开发地址为 `http://localhost:8000`，允许 `http://localhost:3000` 和 `http://localhost:5173` 跨域访问。

协作时从 `main` 创建各自的功能分支，完成后发起 Pull Request；变更接口前先同步 `backend/docs/API.md`。虚拟环境、生成数据和训练产物由各成员本地生成，不提交到仓库。
