# Steel-Billet-OCR (Factory Two-Stage)

本目录是面向工厂上线的二阶段版本，替代原一阶段检测识别流程。

- Stage-1: 整串字符区域 OBB 检测（`models/Stage-1.pt`）
- Stage-2: ROI 字符 OBB 检测 + 排序拼接（`models/Stage-2.pt`）
- API 协议保持与原一阶段一致：`POST /ocr`，请求体仍为 `{"type":"base64","images":"..."}`

## 1. 新工程结构

```text
Steel-Billet-OCR/
  README.md
  requirements.txt
  Dockerfile
  .dockerignore
  LICENSE

  src/
    api.py
    two_stage_engine.py
    __init__.py

  test/scripts/
    check_torch_env.sh
    infer_two_stage.py

  configs/
    yolo11-obb.yaml
    stage2_data.yaml

  models/
    Stage-1.pt
    Stage-2.pt
    data.yaml                # 可选，主要用于健康检查类目统计

  docs/
    DEPLOY_DOCKER.md
    TEST_TUNNEL.md

  test/output/
```

## 2. 快速开始（本地）

```bash
cd /home/yao/TEST/Steel-Billet-OCR
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

批量推理：

```bash
python test/scripts/infer_two_stage.py \
  --stage1 models/Stage-1.pt \
  --stage2 models/Stage-2.pt \
  --source test/images \
  --output test/output \
  --device cpu \
  --save_vis
```

输出：
- `test/output/predictions.csv`
- `test/output/annotated/`（启用 `--save_vis` 时）

## 3. API 协议（保持不变）

- `POST /ocr`
- Header: `Content-Type: application/json`

请求体：

```json
{
  "type": "base64",
  "images": "{图片base64字符串}"
}
```

成功响应：

```json
{"success": true, "result": "1G2065H", "score": 0.91}
```

失败响应：

```json
{"success": false, "errorMsg": "stage1_no_detection"}
```

补充接口：
- `GET /health`
- `GET /metrics`

## 4. 训练与数据转换

当前仓库聚焦部署与推理流程。训练与标注转换脚本将在后续任务中按新结构补充。

## 5. 部署文档

- Docker 工厂部署：`docs/DEPLOY_DOCKER.md`
- 临时公网联调：`docs/TEST_TUNNEL.md`

## 6. 许可

沿用原项目许可，详见 `LICENSE`。
