# 临时公网测试文档（Cloudflare Tunnel）

用于无需对方部署时的临时联调验证。生产请使用 Docker 内网部署。

## 1. 本地服务检查

```bash
curl http://127.0.0.1:8000/health
```

## 2. 启动临时公网通道

```bash
cloudflared tunnel --protocol http2 --url http://127.0.0.1:8000
```

记录输出的临时地址：
- `https://xxxx.trycloudflare.com`

## 3. 接口调用

```bash
curl -X POST https://xxxx.trycloudflare.com/ocr \
  -H "Content-Type: application/json" \
  -d '{"type":"base64","images":"{BASE64}"}'
```

## 4. 结束测试

直接 `Ctrl + C` 停止 cloudflared 即可关闭公网访问。
