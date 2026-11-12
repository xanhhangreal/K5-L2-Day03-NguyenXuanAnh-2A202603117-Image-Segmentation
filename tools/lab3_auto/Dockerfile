FROM cvat.sam.cpu.base:latest
RUN pip3 install --timeout 60 --retries 3 --no-cache-dir ultralytics==8.4.210 transformers==4.57.6 docker==7.2.0 requests==2.34.2
ENV HF_HOME=/cache/huggingface YOLO_CONFIG_DIR=/cache/ultralytics PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
WORKDIR /workspace
