# WebResearcher WebUI

现代化的 Web 用户界面，为 WebResearcher 提供直观的交互体验。

<p align="center">
  <img src="../docs/webui-snap.png" alt="WebUI Screenshot" width="800"/>
</p>

## ✨ 特性

### 🎨 用户界面
- **对话式交互**: 类似 ChatGPT 的对话界面
- **实时流式输出**: 研究过程实时展示
- **三栏布局**: 历史记录 + 对话区 + 研究过程
- **响应式设计**: 完美适配桌面端和移动端

### 📱 移动端适配
- **抽屉式侧边栏**: 从左右两侧滑入的历史记录和研究过程面板
- **触摸优化**: 适合触摸操作的按钮尺寸和间距
- **自适应字体**: 根据屏幕大小自动调整字体
- **流畅动画**: 200ms 平滑过渡效果

### 🔄 研究过程可视化
- **计划展示**: 显示 AI 的分析思路
- **中间报告**: 展示研究进度和发现
- **工具调用**: 实时显示搜索、访问等操作
- **可折叠卡片**: 长内容自动折叠，点击展开

### 📚 历史记录管理
- **会话保存**: 自动保存所有研究会话
- **标题编辑**: 支持修改会话标题
- **会话回放**: 点击历史记录重新加载对话
- **持久化存储**: 数据保存到 `history.jsonl`

### ⚙️ 灵活配置
- **附加指令**: 自定义研究要求（如"优先引用 2023 年后的来源"）
- **工具选择**: 选择特定工具（search, visit, python 等）
- **会话级配置**: 每次研究可独立设置

## 🚀 快速开始

### 1. 安装依赖

```bash
cd webui
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件或设置环境变量：

```bash
# 必需
export LLM_API_KEY="your-openai-or-deepseek-api-key"
export SERPER_API_KEY="your-serper-api-key"

# 可选
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL_NAME="gpt-4o"
```

### 3. 启动服务

```bash
python main.py
```

默认运行在 `http://localhost:8000`

### 4. 访问界面

在浏览器中打开 `http://localhost:8000`，开始使用！

### API 端点

#### 会话管理
- `GET /api/sessions` - 获取所有会话
- `POST /api/session` - 创建新会话
- `DELETE /api/session/{sid}` - 删除会话
- `PUT /api/session/{sid}/title` - 更新会话标题

#### 研究执行
- `POST /api/research` - 开始研究（SSE 流式响应）
- `GET /api/session/{sid}/turn/{tid}/process` - 获取研究过程

#### 数据格式

**历史记录（history.jsonl）：**
```json
{
  "session_id": "abc123",
  "title": "研究主题",
  "created_at": "2025-11-23T10:00:00",
  "updated_at": "2025-11-23T10:05:00",
  "turns": [
    {
      "task_id": "task_001",
      "question": "用户问题",
      "answer": "AI 回答",
      "status": "completed",
      "process": {
        "rounds": [...],
        "tools": [...]
      }
    }
  ]
}
```

## 🔧 配置选项

### 环境变量

```bash
# LLM 配置
LLM_API_KEY=sk-xxx              # LLM API 密钥
LLM_BASE_URL=https://...        # LLM API 地址
LLM_MODEL_NAME=gpt-4o          # 模型名称

# 工具配置
SERPER_API_KEY=xxx             # Google 搜索
JINA_API_KEY=xxx               # 网页抓取（可选）
SANDBOX_FUSION_ENDPOINTS=xxx   # Python 沙箱（可选）

# 服务配置
WEBUI_HOST=0.0.0.0            # 监听地址
WEBUI_PORT=8000               # 监听端口
MAX_LLM_CALL_PER_RUN=50       # 最大迭代次数
FILE_DIR=./files              # 文件目录
```

### 启动参数

```bash
# 自定义端口
python main.py --port 8080

# 监听所有网卡
python main.py --host 0.0.0.0

# 开发模式（自动重载）
uvicorn main:app --reload --port 8000
```

## 🚀 部署

### Docker 部署

```bash
# 构建镜像
docker build -t webresearcher-webui .

# 运行容器
docker run -d \
  -p 8000:8000 \
  -e LLM_API_KEY=your-key \
  -e SERPER_API_KEY=your-key \
  -v $(pwd)/data:/app/data \
  webresearcher-webui
```

### 生产环境部署

使用 Gunicorn + Nginx：

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动应用
gunicorn main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  -b 127.0.0.1:8000

# Nginx 反向代理配置
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

## 📄 许可证

与主项目相同，采用 Apache License 2.0。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
