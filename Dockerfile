FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.10 python3-pip libxrender1 libxext6 && apt-get clean
WORKDIR /workspace
COPY requirements.txt pyproject.toml ./
COPY code ./code
RUN python3.10 -m pip install --no-cache-dir .
ENTRYPOINT ["fedicbs-train"]
