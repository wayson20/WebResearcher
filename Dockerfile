FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY webui/requirements.txt webui/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r webui/requirements.txt

# 复制项目文件
COPY . .

# 创建数据目录
RUN mkdir -p webui/data

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "webui.app:app", "--host", "0.0.0.0", "--port", "8000"]
