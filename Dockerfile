# adding pretrained-models.pytorch support and testing capabilities
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
#install additional dependencies
RUN apt-get update && apt-get install -y wget graphviz
RUN conda install -y -c conda-forge -c sbugallo protobuf numpy matplotlib scipy pandas jupyter "pylint<2.0.0" rope && \
    conda clean --all && \
    pip install tensorboard onnx onnx-simplifier && \
    pip install onnxruntime-gpu pydot && \
    conda install -y -c conda-forge -c sbugallo ffmpeg py-opencv==4.2.0 && \
    conda clean --all

RUN apt-get update && apt-get install -y libgl1-mesa-glx

#RUN dpkg -i nv-tensorrt-repo-ubuntu1x04-cudax.x-trt5.x.x.x-ga-yyyymmdd_1–1_amd64.deb
#RUN apt-key add /var/nv-tensorrt-repo-cudax.x-trt5.x.x.x-ga-yyyymmdd/7fa2af80.pub
#RUN apt-get update
#RUN apt-get install tensorrt 
#RUN apt-get install uff-converter-tf

#RUN mkdir /workspace/tmp/ && cd /workspace/tmp/ && \
#    git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
#    cd /workspace/tmp/torch2trt && \
#    python setup.py install

#switch to non-root user with sudo privileges
RUN apt-get update && apt-get install sudo && \
    adduser --disabled-password --gecos "" udocker && \
    adduser udocker sudo && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER udocker
RUN sudo chown -R udocker:udocker /workspace/
ENV PATH="/home/udocker/.local/bin:${PATH}"

#starting jupyter notebook with password "pyt"
CMD /bin/bash -c "jupyter notebook --ip=0.0.0.0 --port=8889 --allow-root --no-browser --NotebookApp.password='sha1:32ee031c72e3:747ead40b08c0692a23dd025172734e3f4f3ce11'"

#run container with data added volume: --runtime=nvidia --shm-size 8G -v /path/to/data:/workspace/data:delegated
#add -v /path/to/torch_tmp/:/root/.torch:delegated to prevent redownloading of base models
#add -e NVIDIA_VISIBLE_DEVICES=2,3 to limit used nvidia devices





