# Mesh node image. Every role uses it:
#   docker run -v $PWD/mesh-data:/data -p 9100:9100 mesh node --dir /data --name validator0
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mesh mesh
COPY swarm swarm
RUN useradd --create-home mesh && mkdir /data && chown mesh /data
USER mesh
ENV OPENBLAS_NUM_THREADS=1
VOLUME /data
ENTRYPOINT ["python", "-m", "mesh.run"]
