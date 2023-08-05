import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from collections import OrderedDict
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology

import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import mvtec as mvtec

import cv2

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# video capture and size
cap = cv2.VideoCapture(0)
size = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='E:/Github_Repos/Defect-Detection-With-Robotic-Arm/roundCNN/PADIM/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()


def main():
    args = parse_args()

    model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
    t_d = 1792
    d = 550

    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)

     #  train outputs load
    train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % "bottle")
    print('load train set feature from: %s' % train_feature_filepath)
    with open(train_feature_filepath, 'rb') as f:
        train_outputs = pickle.load(f)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)

    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

     # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)


    while True:
        # Getting frame
        ret, fframe = cap.read()

        # Resizing the frame
        resized_frame = cv2.resize(fframe, size)  

        a = cv2.imread("E:/Github_Repos/Defect-Detection-With-Robotic-Arm/roundCNN/PADIM/mvtec_anomaly_detection/bottle/test/broken_small/009.png")
        resized_frame = cv2.resize(a, size)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (((resized_frame * std) + mean) * 255.).astype(np.uint8)

        frame_transposed = normalized.transpose((2, 0, 1))  # Channel-last -> Channel-first
    
        x = np.expand_dims(frame_transposed, axis=0)

        x = torch.from_numpy(x).float()
       

        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    
        # model prediction
        with torch.no_grad():
            _ = model(x.to(device))

        # get intermediate layer outputs
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.cpu().detach())

        # initialize hook outputs
        outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                    align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        scores = np.expand_dims(scores, axis=0)

        
        new_mask = plot_fig(scores, 0.3)

        edged = cv2.Canny(np.uint8(new_mask), 0, 255)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(resized_frame, contours, -1, (0, 255, 0), 3)
        cv2.imshow('Contours', resized_frame)
        cv2.imwrite("YUZLESME.jpg", resized_frame)
        cv2.imwrite("YUZLESME_MASK.jpg", new_mask)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def plot_fig(scores, threshold):
    num = len(scores)

    for i in range(num):
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255

        return mask
    


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x * std) + mean) * 255.).astype(np.uint8)
    
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    main()
    cap.release()
    cv2.destroyAllWindows()
