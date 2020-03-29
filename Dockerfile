# adding pretrained-models.pytorch support and testing capabilities
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
#install additional dependencies
RUN apt-get update && apt-get install -y wget
RUN conda install -y matplotlib scipy pandas jupyter "pylint<2.0.0" rope && conda clean --all && pip install tensorboardX
#necessary for scipy legacy imread function
RUN /usr/bin/yes | pip uninstall scipy
RUN /usr/bin/yes | pip install scipy==1.1.0
RUN /usr/bin/yes | pip install tensorboard
RUN apt-get update && apt-get install -y graphviz

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





