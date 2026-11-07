# One build stage exports the weights with torch; the runtime image never contains torch.
# It writes both UNet formats, so KSHANA_UNET_BACKEND can switch without a rebuild.
#   docker build -t kshana . && docker run -p 8000:8000 kshana
# Modal builds this file via modal.Image.from_dockerfile (server/modal_app.py).

FROM python:3.11-slim AS export
WORKDIR /build
# CPU torch first: accelerate would otherwise pull the much larger default build.
RUN pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir openvino==2026.4.1 onnxruntime==1.31.1 onnx==1.23.2 numpy==2.2.6 "tokenizers>=0.21,<1" diffusers==0.42.1 transformers==5.22.1 safetensors accelerate timm
RUN opt_in_out --opt_out
COPY optimize/export_openvino.py optimize/export_openvino.py
COPY kshana/__init__.py kshana/generator.py kshana/ort.py kshana/
# Downloads the weights from Hugging Face and fails the build if the IR check errors.
RUN PYTHONPATH=. python optimize/export_openvino.py --out /models --backends onnxruntime
# The output safety gate needs this model, or every image would be refused.
COPY optimize/export_nsfw.py optimize/export_nsfw.py
RUN PYTHONPATH=. python optimize/export_nsfw.py --onnx /models/nsfw.onnx

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt
# OpenVINO otherwise sends analytics on import.
RUN opt_in_out --opt_out
COPY --from=export /models /models
COPY kshana/ kshana/
COPY server/__init__.py server/app.py server/jobs.py server/
COPY web/ web/
ENV PYTHONPATH=/app KSHANA_MODELS=/models KSHANA_THREADS=4 OMP_NUM_THREADS=4
CMD ["uvicorn", "server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
