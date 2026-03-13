# Steel-Billet-OCR

面向工厂现场的钢坯号二阶段 OCR 服务。工程职责只有两件事：

- 提供 `POST /ocr` HTTP 接口
- 提供本地批量推理与结果可视化能力

当前仓库聚焦推理与部署，不包含完整训练流水线。

## 1. 识别路线

当前识别流程是标准二阶段 OBB 路线：

1. Stage-1 检测整串钢坯号区域
2. 对该区域做透视变换，拉正为 ROI
3. Stage-2 在 ROI 内检测单字符
4. 按字符串主轴方向排序字符
5. 将字符标签直接拼接成最终结果

核心实现位于：

- [src/api.py](/home/yao/TEST/Steel-Billet-OCR/src/api.py)
- [src/two_stage_engine.py](/home/yao/TEST/Steel-Billet-OCR/src/two_stage_engine.py)

## 2. 工程结构

```text
Steel-Billet-OCR/
├── README.md
├── LICENSE
├── Dockerfile
├── requirements.txt
├── configs/
│   ├── data-frame.yaml        # Stage-1 类目配置
│   ├── data-char.yaml         # Stage-2 字符类目配置
│   └── yolo11-obb.yaml        # YOLO11-OBB 结构配置
├── docs/
│   ├── DEPLOY_DOCKER.md
│   └── TEST_TUNNEL.md
├── models/
│   ├── stage-1/
│   │   └── README.md
│   └── stage-2/
│       └── README.md
├── src/
│   ├── __init__.py
│   ├── api.py
│   └── two_stage_engine.py
└── test/
    ├── images/               # 示例测试图片
    ├── output/               # 推理输出目录
    └── scripts/
        ├── check_torch_env.sh
        └── infer_two_stage.py
```

## 3. 模型文件说明

模型权重默认不直接提交到 Git 仓库，模型下载地址记录在：

- [models/stage-1/README.md](/home/yao/TEST/Steel-Billet-OCR/models/stage-1/README.md)
- [models/stage-2/README.md](/home/yao/TEST/Steel-Billet-OCR/models/stage-2/README.md)

当前目录约定如下：

- `models/stage-1/Stage-1-S-base.pt`
- `models/stage-1/Stage-1-X-Advanced.pt`
- `models/stage-2/Stage-2-S-base.pt`
- `models/stage-2/Stage-2-S-base-fine-tune.pt`
- `models/stage-2/Stage-2-X-Advanced.pt`

工厂部署时至少需要准备：

- 1 个 Stage-1 模型
- 1 个 Stage-2 模型

推荐起步组合：

- Stage-1: `models/stage-1/Stage-1-S-base.pt`
- Stage-2: `models/stage-2/Stage-2-S-base.pt`

## 4. 运行环境

建议环境：

- Python 3.10
- Linux
- CPU 部署可直接使用当前 Dockerfile
- GPU 部署需自行调整 PyTorch 与运行参数

安装依赖：

```bash
cd /home/yao/TEST/Steel-Billet-OCR
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

快速检查 Torch 环境：

```bash
bash test/scripts/check_torch_env.sh
```

## 5. 本地批量推理

使用测试脚本对目录下图片批量推理：

```bash
python test/scripts/infer_two_stage.py \
  --stage1 models/stage-1/Stage-1-S-base.pt \
  --stage2 models/stage-2/Stage-2-S-base.pt \
  --source test/images \
  --output test/output \
  --device cpu \
  --save_vis
```

输出内容：

- `test/output/predictions.csv`
- `test/output/annotated/`，仅在传入 `--save_vis` 时生成

脚本作用：

- 读取 `test/images/` 下图片
- 对每张图片执行两阶段识别
- 导出识别结果 CSV
- 可选保存带框可视化图片

## 6. API 服务

### 6.1 启动方式

本地直接启动：

```bash
export STAGE1_MODEL=/home/yao/TEST/Steel-Billet-OCR/models/stage-1/Stage-1-S-base.pt
export STAGE2_MODEL=/home/yao/TEST/Steel-Billet-OCR/models/stage-2/Stage-2-S-base.pt
export DATA_YAML=/home/yao/TEST/Steel-Billet-OCR/configs/data-char.yaml
export DEVICE=cpu
export CONF1=0.25
export CONF2=0.50
export IOU=0.70
export EXPAND1=1.08
export PAD=0.10

uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 6.2 接口规范

主接口：

- `POST /ocr`
- `Content-Type: application/json`

请求体：

```json
{
  "type": "base64",
  "images": "{图片base64字符串}"
}
```

成功响应示例：

```json
{
  "success": true,
  "result": "1G2065H",
  "score": 0.91
}
```

失败响应示例：

```json
{
  "success": false,
  "errorMsg": "stage1_no_detection"
}
```

辅助接口：

- `GET /health`
- `GET /metrics`

### 6.3 常见失败状态

接口可能返回以下 `errorMsg`：

- `decode_error`：请求图片 base64 解码失败
- `read_failed`：图像读取失败
- `stage1_no_detection`：未找到整串钢坯号区域
- `warp_failed`：透视变换失败
- `stage2_no_detection`：Stage-2 未检测到字符

## 7. Docker 部署

当前仓库提供 CPU 版 Dockerfile。
由于 Git 仓库默认不包含模型权重，生产部署建议使用宿主机目录挂载模型文件，不建议依赖镜像内置权重。

构建镜像：

```bash
cd /home/yao/TEST/Steel-Billet-OCR
docker build -t steel-billet-ocr:2stage-cpu .
```

推荐生产运行方式：

```bash
docker run -d --name steel-billet-ocr \
  -p 8000:8000 \
  -v /data/steel-billet-models:/app/models:ro \
  -e STAGE1_MODEL=/app/models/stage-1/Stage-1-S-base.pt \
  -e STAGE2_MODEL=/app/models/stage-2/Stage-2-S-base.pt \
  -e DATA_YAML=/app/configs/data-char.yaml \
  -e DEVICE=cpu \
  -e CONF1=0.25 \
  -e CONF2=0.50 \
  -e IOU=0.7 \
  -e EXPAND1=1.08 \
  -e PAD=0.10 \
  steel-billet-ocr:2stage-cpu
```

模型目录示例：

- `/data/steel-billet-models/stage-1/Stage-1-S-base.pt`
- `/data/steel-billet-models/stage-2/Stage-2-S-base.pt`

如果你确实已经把模型打进镜像，也可以不挂载模型目录：

```bash
docker run -d --name steel-billet-ocr \
  -p 8000:8000 \
  -v /data/steel-billet-models:/app/models \
  -e STAGE1_MODEL=/app/models/stage-1/Stage-1-S-base.pt \
  -e STAGE2_MODEL=/app/models/stage-2/Stage-2-S-base.pt \
  -e DATA_YAML=/app/configs/data-char.yaml \
  steel-billet-ocr:2stage-cpu
```

更详细的部署说明见：

- [docs/DEPLOY_DOCKER.md](/home/yao/TEST/Steel-Billet-OCR/docs/DEPLOY_DOCKER.md)
- [docs/TEST_TUNNEL.md](/home/yao/TEST/Steel-Billet-OCR/docs/TEST_TUNNEL.md)

## 8. 测试资源说明

仓库中保留了少量现场测试图片，位于：

- [test/images](/home/yao/TEST/Steel-Billet-OCR/test/images)

输出目录位于：

- [test/output](/home/yao/TEST/Steel-Billet-OCR/test/output)

说明：

- `test/images/` 用于本地快速回归验证
- `test/output/` 用于存放推理结果，不建议提交大批量生成文件

## 9. 投产前检查清单

建议在工厂上线前逐项确认：

- Stage-1 与 Stage-2 模型文件均已放到约定目录
- `DATA_YAML` 指向 `configs/data-char.yaml`
- `GET /health` 返回 `success: true`
- 通过 `test/images/` 或现场样图完成一次批量推理
- 抽查 `predictions.csv` 与可视化结果，确认字符顺序正确
- 使用真实调用方报文完成一次 `POST /ocr` 联调
- 确认端口、防火墙、日志采集与监控抓取正常

## 10. 关键参数说明

服务和脚本共用的主要参数如下：

- `STAGE1_MODEL`：Stage-1 模型路径
- `STAGE2_MODEL`：Stage-2 模型路径
- `DATA_YAML`：Stage-2 字符类别配置路径
- `DEVICE`：`cpu` 或 GPU 设备号
- `CONF1`：Stage-1 置信度阈值
- `CONF2`：Stage-2 置信度阈值
- `IOU`：NMS IoU 阈值
- `EXPAND1`：Stage-1 框外扩比例
- `PAD`：透视变换时 ROI 边缘补白比例

## 11. 当前边界

当前仓库已覆盖：

- 双阶段推理
- HTTP 服务
- 批量测试脚本
- Docker CPU 部署

当前仓库未覆盖：

- 完整训练脚本
- 数据转换脚本


## 12. 许可

本项目采用仓库内的专有非商业许可协议，完整条款见 [`LICENSE`](./LICENSE)。

该许可不是开源许可证，作者保留所有权利。除非你已经取得作者事先书面授权，否则本项目仅可用于非商业测试、评估、教学、课程实验、学术研究和有限范围内的学术交流。

以下行为默认被明确禁止：

- 将本项目用于任何商业用途，包括生产部署、业务系统上线、商业项目交付、招投标交付或盈利性运营
- 以 API、SaaS、托管服务、远程调用或其他方式向第三方提供本项目能力
- 传播、公开发布、共享、镜像、转让、出租或向第三方提供源码、模型、权重、Docker 镜像、部署包或其他可复现材料
- 修改、改编、重构、翻译、制作衍生作品，或对本项目进行再发布、再分发、再许可

需要特别注意：

- 即使不收费，只要用于商业组织、生产场景、业务交付或第三方服务，也可能构成商业用途
- 第三方依赖组件仍受其各自许可证约束，本仓库许可不替代也不改变第三方组件的授权条款
- 如需商业使用、部署授权或其他超出本许可范围的用途，必须先按 [`LICENSE`](./LICENSE) 中的联系方式向作者申请书面授权
