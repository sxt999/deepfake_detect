"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
"""
import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
import numpy as np

from network.models import model_selection
from dataset.transform import xception_default_data_transforms, transforms_380
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import torchvision.transforms as transforms
from xception import xception

# class BaseDataset(Dataset):
#     def __init__(self, root, transform=None, num_classes=2):
#         super(BaseDataset,self).__init__()
#         self.root = root
#         self.transform = transform
#         self.num_classes = num_classes
#         # assert transform is not None, "transform is None"

#     def __getitem__(self,idx):
#         print("---------------")
#         img_path = self.imgs[idx][0]
#         print(img_path)
#         label = self.imgs[idx][1]

#         image = pil_image.open(img_path).convert('RGB')
#         image = self.transform(image)
#         return (image, img_path)

#     def __len__(self):
#         return len(self.imgs)

# class trainCelebDF(BaseDataset):
#     def __init__(self,root,train_type="train",transform=None,num_classes=2):
#         super(trainCelebDF,self).__init__(root=root, transform=transform, num_classes=num_classes)

#         print(" in  dataset ")
#         real_root = "/home/adminadmin/deepfake_detect/Deepfake-Detection-master/images/"
#         synthesis_root = "/home/adminadmin/deepfake_detect/Deepfake-Detection-master/images1/"

#         real_imgs = []
#         fake_imgs = []

#         real_imgs = glob.glob(os.path.join(real_root, "*.png"))
#         fake_imgs = glob.glob(os.path.join(synthesis_root, "*.png"))

#         print("[{}]\t fake imgs count :{}, real imgs count :{}".format(train_type, len(fake_imgs),len(real_imgs)))
#         fake_imgs = [[p,1] for p in fake_imgs]
#         real_imgs = [[p,0] for p in real_imgs]
#         self.imgs = fake_imgs + real_imgs

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, cuda=True, SelfBlend=False):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    if SelfBlend is False:
        preprocess = xception_default_data_transforms['test']
    else:
        preprocess = transforms_380['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True, SelfBlend=False):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda, SelfBlend)

    # Model prediction
    output = model(preprocessed_image)

    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output

def predict_with_model_CORE(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True, SelfBlend=False):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda, SelfBlend)

    # Model prediction
    # model.eval()

    _, output = model(preprocessed_image)

    output = post_function(output)

    # Cast to desired

    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output

def predict_with_model_SelfBlend(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True, SelfBlend=False):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda, SelfBlend)

    # Model prediction
    # model.eval()

    output = model(preprocessed_image)
    print(output)
    output = post_function(output)
    print(output)
    # Cast to desired

    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output

# def get_augs(name="base", norm="imagenet", size=299):
#     IMG_SIZE = size
#     if norm == "imagenet":
#         mean = [0.485, 0.456, 0.406]
#         std = [0.229, 0.224, 0.225]
#     elif norm == "0.5":
#         mean = [0.5, 0.5, 0.5]
#         std = [0.5, 0.5, 0.5]
#     else:
#         mean = [0, 0, 0]
#         std = [1, 1, 1]

#     if name == "None":
#         return transforms.Compose([
#             transforms.Resize(IMG_SIZE),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean,std=std),
#         ])

def test_full_image_network(image_path, model_name, output_path, cuda=True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored

    :param cuda: enable cuda
    :return:
    """
    print('Starting: {}'.format(image_path))
    docker_root = "/df_detect/"

    # Read and write

    os.makedirs(output_path, exist_ok=True)
    if os.path.isdir(image_path) is False:
        imgs_paths = [image_path]
    else:
        imgs_paths = glob.glob(os.path.join(image_path, "*"))
    imgs = []
    for img_path in imgs_paths:
        image = pil_image.open(img_path).convert('RGB')
        image = np.array(image)
        imgs.append((image, img_path))

    # test_augs = get_augs(name="None",norm="0.5",size=299)
    # test_dataset = trainCelebDF("root","test",test_augs)
    # testloader = DataLoader(test_dataset,
    #     batch_size = 1,
    #     shuffle = True,
    #     num_workers = 0
    # )
    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Load model
    if model_name == "Base":
        model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    elif model_name == "CORE":
        model = model_selection(modelname='CORE', num_out_classes=2, dropout=0.5)
    elif model_name == "SelfBlend":
        model = model_selection(modelname='SelfBlend', num_out_classes=2, dropout=0.5)
    else:
        print("fatal: availble model: Base, CORE, SelfBlend")
        return
    if model_name == "Base":
        model.load_state_dict(torch.load(os.path.join(docker_root, "pretrained_model/xception/best.pkl"), map_location="cpu"))
    elif model_name == "CORE":
        model.load_state_dict(torch.load(os.path.join(docker_root, "pretrained_model/CORE/epoch_16_acc_98.929_auc_99.549.pth"), map_location="cpu")['model'])
        model.eval()
    elif model_name == "SelfBlend":
        model.load_state_dict(torch.load(os.path.join(docker_root, "pretrained_model/SelfBlend/epoch_2_acc_100.000_3.pth"), map_location="cpu")['model'])
        model.eval()
    else:
        print("fatal: availble model: Base, CORE, SelfBlend")
        return
    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module
    # if cuda:
    #     model = model.cuda()

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1


    pbar = tqdm(total=len(imgs))

    for batch_idx,(image,img_path) in enumerate(imgs): 
        print(img_path)
        # print(image.shape)
        video_fn = img_path.split('/')[-1]
        if image is None:
            break

        pbar.update(1)

        # Image size
        height, width = image.shape[:2]


        # 2. Detect with dlib
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(join("./test2", video_fn), image)
        faces = face_detector(gray, 1)
        # faces = [image]
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]


            # Actual prediction using our model
            SelfBlend = False
            if model_name == "SelfBlend":
                SelfBlend = True
            if model_name == "Base":
                prediction, output = predict_with_model(cropped_face, model,
                                                        cuda=cuda, SelfBlend=SelfBlend)
            elif model_name == "CORE":
                prediction, output = predict_with_model_CORE(cropped_face, model,
                                                        cuda=cuda, SelfBlend=SelfBlend)
            elif model_name == "SelfBlend":
                prediction, output = predict_with_model_SelfBlend(cropped_face, model,
                                                        cuda=cuda, SelfBlend=SelfBlend)
            # ------------------------------------------------------------------

            # Text and bb
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            label = 'fake' if prediction == 1 else 'real'
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            output_list = ['{0:.2f}'.format(float(x)) for x in
                           output.detach().cpu().numpy()[0]]
            # print(output_list)
            # print(label)
            # continue
            cv2.putText(image, label, (x, y+h+30),
                        font_face, font_scale,
                        color, thickness, 2)
            # draw box over face
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)



        # Show

        image = pil_image.fromarray(image)
        video_fn1 = video_fn.split(".")
        # print(video_fn1)
        if prediction:
            video_fn = video_fn1[0] + "_fake." + video_fn1[1]
        else:
            video_fn = video_fn1[0] + "_real." + video_fn1[1]
        image.save(join(output_path, video_fn))

    pbar.close()



if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--image_path', '-i', type=str)
    # p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--model_name', '-mi', type=str, default="Base")
    p.add_argument('--output_path', '-o', type=str,
                   default='.')
    # p.add_argument('--start_frame', type=int, default=0)
    # p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()


    test_full_image_network(**vars(args))
    # video_path = args.video_path
    # if os.path.isdir(video_path) is False:
    #     test_full_image_network(**vars(args))
    # else:
    #     videos = os.listdir(video_path)
    #     for video in videos:
    #         args.video_path = join(video_path, video)
    #         test_full_image_network(**vars(args))