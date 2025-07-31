# Use NVIDIA PyTorch container as base image
FROM nvcr.io/nvidia/pytorch:25.04-py3
ARG TARGETPLATFORM

# Install basic tools
RUN apt-get update && apt-get install -y git tree ffmpeg wget
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN if [[ ${TARGETPLATFORM} == 'linux/amd64' ]]; then ln -s /lib64/libcuda.so.1 /lib64/libcuda.so; fi
RUN apt-get install -y libglib2.0-0
RUN sed -i -e 's/h11==0.14.0/h11==0.16.0/g' /etc/pip/constraint.txt

# Install Flash Attention 3
RUN MAX_JOBS=$(( $(nproc) / 4 )) pip install git+https://github.com/Dao-AILab/flash-attention.git@27f501d#subdirectory=hopper
COPY cosmos_predict2/utils/flash_attn_3/flash_attn_interface.py /usr/local/lib/python3.12/dist-packages/flash_attn_3/flash_attn_interface.py
COPY cosmos_predict2/utils/flash_attn_3/te_attn.diff /tmp/te_attn.diff
RUN patch /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention.py /tmp/te_attn.diff

# Installing decord from source on ARM
COPY Video_Codec_SDK_13.0.19.zip* /workspace/Video_Codec_SDK_13.0.19.zip
RUN if [[ ${TARGETPLATFORM} == 'linux/arm64' ]]; then export DEBIAN_FRONTEND=noninteractive && \
apt-get update && \
apt-get install -y build-essential python3-dev python3-setuptools make cmake \
                   ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev git ssh unzip nano python3-pip && \
git clone --recursive https://github.com/dmlc/decord && \
cd decord && \
find . -type f -exec sed -i "s/AVInputFormat \*/const AVInputFormat \*/g" {} \; && \
sed -i "s/[[:space:]]AVCodec \*dec/const AVCodec \*dec/" src/video/video_reader.cc && \
sed -i "s/avcodec\.h>/avcodec\.h>\n#include <libavcodec\/bsf\.h>/" src/video/ffmpeg/ffmpeg_common.h && \
mkdir build && cd build && \
scp /workspace/Video_Codec_SDK_13.0.19.zip . && \
unzip Video_Codec_SDK_13.0.19.zip && \
cp Video_Codec_SDK_13.0.19/Lib/linux/stubs/aarch64/* /usr/local/cuda/lib64/ && \
cp Video_Codec_SDK_13.0.19/Interface/* /usr/local/cuda/include && \
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && \
make -j 4 && \
cd ../python && python3 setup.py install; fi
RUN if [[ ${TARGETPLATFORM} == 'linux/arm64' ]]; then apt remove -y python3-blinker; fi

# Install the dependencies from requirements-docker.txt
COPY ./requirements-docker.txt /requirements.txt
ARG NATTEN_CUDA_ARCH="8.0;8.6;8.9;9.0;10.0;10.3;12.0"
RUN pip install --no-cache-dir -r /requirements.txt
RUN mkdir -p /workspace
WORKDIR /workspace


#Install the other required packages
COPY ./requirements-fast_api.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN mkdir -p /workspace

WORKDIR /workspace

#download the cosmos_predict 2 from the git repo and copy all files to the workspace
#remove the previous repo if it exists
RUN rm -rf /workspace/cosmos-predict2
RUN git clone https://github.com/samiazirar/cosmos-predict2.git 
RUN mv /workspace/cosmos-predict2/* /workspace/
RUN mv /workspace/cosmos-predict2/.git /workspace/
RUN rm -rf /workspace/cosmos-predict2
#copy all files at once to the workspace from the pulled cosmos_predict2 repo
# RUN cp -r /workspace/cosmos-predict2/* /workspace/
#start from the repo
#your root is the folder where the repo is cloned in 




CMD ["/bin/bash"]

