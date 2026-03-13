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
- ~~GPU 部署需自行调整 PyTorch 与运行参数~~

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
export CONF2=0.55
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

构建镜像：

```bash
cd /home/yao/TEST/Steel-Billet-OCR
docker build -t steel-billet-ocr:2stage-cpu .
```

运行容器时请显式指定模型路径：

```bash
docker run -d --name steel-billet-ocr \
  -p 8000:8000 \
  -e STAGE1_MODEL=/app/models/stage-1/Stage-1-S-base.pt \
  -e STAGE2_MODEL=/app/models/stage-2/Stage-2-S-base.pt \
  -e DATA_YAML=/app/configs/data-char.yaml \
  -e DEVICE=cpu \
  -e CONF1=0.25 \
  -e CONF2=0.55 \
  -e IOU=0.7 \
  -e EXPAND1=1.08 \
  -e PAD=0.10 \
  steel-billet-ocr:2stage-cpu
```

如果镜像内没有模型，请自行挂载模型目录，例如：

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

## 9. 关键参数说明

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

## 10. 当前边界

当前仓库已覆盖：

- 双阶段推理
- HTTP 服务
- 批量测试脚本
- Docker CPU 部署

当前仓库未覆盖：

- 完整训练脚本
- 数据转换脚本
- 自动化单元测试
- 字符规则纠错与后处理模板

## 11. 许可

许可证见 [LICENSE](/home/yao/TEST/Steel-Billet-OCR/LICENSE)。
