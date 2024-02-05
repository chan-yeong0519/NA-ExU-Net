FROM nvcr.io/nvidia/pytorch:21.04-py3

RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    wget

RUN wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz && \
    tar -xvf Python-3.6.9.tgz && \
    cd Python-3.6.9 && \
    ./configure && \
    make && \
    make install

RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.6 1

RUN pip3 install wrapt --upgrade --ignore-installed

RUN pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install wandb --upgrade
RUN wandb login 398ae091e9c781c2974ce0d7c22e09d953920e16

RUN pip install scikit-learn
RUN pip install scipy
#RUN pip install pytorch_lightning
RUN pip install omegaconf==2.0.6
RUN pip install torchsummary
RUN pip install torch_audiomentations
