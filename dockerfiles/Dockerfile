FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04


RUN apt-get -y update && \
    apt-get -y install python3

# RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y cmake
RUN apt-get -y install python3-pip
RUN apt-get install -y libopencv-dev
RUN apt-get install -y git

# RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install torch torchvision torchaudio
RUN pip3 install dlib imutils scipy pandas opencv-python tqdm pretrainedmodels imgaug efficientnet_pytorch facenet_pytorch albumentations -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -U retinaface_pytorch -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt-get clean

WORKDIR /df_detect
