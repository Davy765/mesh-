# Mesh node image. Every role uses it:
#   docker run -v $PWD/mesh-data:/data -p 9100:9100 mesh node --dir /data --name validator0
FROM python:3.15-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mesh mesh
COPY swarm swarm
RUN useradd --create-home mesh && mkdir /data && chown mesh /data
USER mesh
ENV OPENBLAS_NUM_THREADS=1
VOLUME /data
# 9100: node-to-node traffic. 8000: the playground web page (mesh.run app, or --ui-port 8000)
EXPOSE 9100 8000
ENTRYPOINT ["python", "-m", "mesh.run"]
