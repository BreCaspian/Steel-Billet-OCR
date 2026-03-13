# 工厂部署指南（Docker CPU，二阶段）

本指南用于工厂现场部署二阶段识别服务（Stage-1 + Stage-2）。
通信协议保持不变，仍为 `POST /ocr` 且请求体为 `{"type":"base64","images":"..."}`。

## 1. 接口规范

服务地址：
- `http://{ip}:{port}/ocr`

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

## 2. 构建前检查

请先准备模型文件。模型默认不直接提交到 Git 仓库，可按以下目录放置：
- `models/stage-1/Stage-1-S-base.pt`
- `models/stage-2/Stage-2-S-base.pt`

可选：
- `configs/data-char.yaml`

## 3. 构建镜像

```bash
cd /home/yao/TEST/Steel-Billet-OCR
docker build -t steel-billet-ocr:2stage-cpu .
```

## 4. 启动服务

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

## 5. 健康检查与调用

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/ocr \
  -H "Content-Type: application/json" \
  -d '{"type":"base64","images":"{BASE64}"}'
```

## 6. 监控与日志

- 健康检查：`GET /health`
- 指标采集：`GET /metrics`

关键指标：
- `ocr_requests_total{status="ok|decode_error|stage1_no_detection|warp_failed|stage2_no_detection|read_failed"}`
- `ocr_request_latency_seconds`
- `ocr_detections_total{result="ok"}`

查看日志：

```bash
docker logs -f steel-billet-ocr
```

## 7. 离线镜像部署

有网机器：

```bash
docker build -t steel-billet-ocr:2stage-cpu .
docker save steel-billet-ocr:2stage-cpu | gzip > steel-billet-ocr_2stage_cpu.tar.gz
```

工厂机器：

```bash
gunzip -c steel-billet-ocr_2stage_cpu.tar.gz | docker load
docker run -d --name steel-billet-ocr \
  -p 8000:8000 \
  -e STAGE1_MODEL=/app/models/stage-1/Stage-1-S-base.pt \
  -e STAGE2_MODEL=/app/models/stage-2/Stage-2-S-base.pt \
  -e DATA_YAML=/app/configs/data-char.yaml \
  steel-billet-ocr:2stage-cpu
```
