import random
from random import sample
import argparse
import numpy as np
import os
import h5py
from collections import OrderedDict
from scipy.ndimage import gaussian_filter
from skimage import morphology

import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, resnet18, efficientnet_v2_s
import torchvision.transforms as T

import timm
import cv2
from PIL import Image


# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# video capture and size
cap = cv2.VideoCapture(0)
size = (224, 224)

cv2.namedWindow('output')
threshold = np.float32(0.3)

def nothing(trash):
    pass

cv2.createTrackbar("Threshold", 'output', 55, 100, nothing)


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('-d', '--data_path', type=str, default='E:/Github_Repos/Defect-Detection-With-Robotic-Arm/roundCNN/PADIM/mvtec_anomaly_detection')
    parser.add_argument('-s', '--save_path', type=str, default='./mvtec_result')
    parser.add_argument('-a', '--arch', type=str, choices=['resnet18', 'wide_resnet50_2',
     'efficientnetv2_m_in21ft1k', 'efficientnetv2_xl_in21ft1k', 
     'efficientnet_b5_ns', 'efficientnet_b6_ns', 'efficientnet_b7_ns', 
     'efficientnet_l2_ns_475'], default='efficientnetv2_m_in21ft1k')
    parser.add_argument('-r', '--reduce_dim', action='store_true')
    parser.add_argument('-p', '--pca', action="store_true", help="Enable pca")
    parser.add_argument('-n', '--npca', action="store_true", help="Enable npca")
    parser.add_argument('-v', '--variance_threshold', type=float, default=0.99, help="Variance threshold to apply")
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--save_gpu_memory', action='store_true', help='In case of gpu OOM')
    # parser.add_argument('-c', '--class_name',)
    return parser.parse_args()  


def main():
    global threshold

    args = parse_args()

    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    elif args.arch == 'efficientnetv2_m_in21ft1k':
        model = timm.create_model('tf_efficientnetv2_m_in21ft1k', pretrained=True)
        t_d = (80 + 160 + 304) # (48 + 80 + 160 + 176 + 304) features1,2,3,4,5
        d = 100
    elif args.arch == 'efficientnetv2_xl_in21ft1k':
        model = timm.create_model('tf_efficientnetv2_xl_in21ft1k', pretrained=True)
        t_d = (192 + 256 + 512) # (64 + 96 + 192 + 256 + 512) features1,2,3,4,5
        d = 100
    elif args.arch == 'efficientnet_b5_ns':
        model = timm.create_model('tf_efficientnet_b5_ns', pretrained=True)
        t_d = (40 + 64 + 176) # (40 + 64 + 128 + 176 + 304) features1,2,3,4,5
        d = 100
    elif args.arch == 'efficientnet_b6_ns':
        model = timm.create_model('tf_efficientnet_b6_ns', pretrained=True)
        t_d = (40 + 72 + 200) # (40 + 72 + 144 + 200 + 344) features1,2,3,4,5
        d = 100
    elif args.arch == 'efficientnet_b7_ns':
        model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True)
        t_d = (48 + 80 + 224) # (48 + 80 + 160 + 224 + 384) features1,2,3,4,5
        d = 100
    elif args.arch == 'efficientnet_l2_ns_475':
        model = timm.create_model('tf_efficientnet_l2_ns_475', pretrained=True)
        t_d = (104 + 344 + 480) # (104 + 176 + 344 + 480 + 824) features1,2,3,4,5
        d = 550

    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    class_name = "metal_nut"

# train output load
    train_feature_filepath = os.path.join(args.save_path, f'temp_{args.arch}', f'train_{class_name}.hdf5')
    
    print(f'load train set feature from: {train_feature_filepath}')
    with h5py.File(train_feature_filepath, 'r') as f:
        train_outputs = [f['mean'][()], f['cov_inv'][()]]
            
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    if args.reduce_dim:
        idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    if 'resnet' in args.arch:
        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)
    elif args.arch == 'efficientnetv2_m_in21ft1k':
        model.blocks[2][-1].register_forward_hook(hook)
        model.blocks[3][-1].register_forward_hook(hook)
        model.blocks[5][-1].register_forward_hook(hook)
    elif args.arch == 'efficientnet_b7_ns':
        model.blocks[1][-1].register_forward_hook(hook)
        model.blocks[3][-1].register_forward_hook(hook)
        model.blocks[4][-1].register_forward_hook(hook)
    elif 'efficientnet' in args.arch:
        model.blocks[1][-1].register_forward_hook(hook)
        # model.blocks[2][-1].register_forward_hook(hook)
        model.blocks[3][-1].register_forward_hook(hook)
        model.blocks[4][-1].register_forward_hook(hook)
        # model.blocks[2][0].register_forward_hook(hook)
        # model.blocks[3][0].register_forward_hook(hook)
        # model.blocks[4][0].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)

    
    use_gpu = args.use_gpu

    while True:
        ret, fframe = cap.read()  # Getting frame

        mean=[0.5, 0.5, 0.5]
        std=[0.5, 0.5, 0.5]
        
        compose = T.Compose([T.ToPILImage(),
                             T.Resize(256, Image.ANTIALIAS),
                             T.CenterCrop(224),
                             T.ToTensor(),
                             T.Normalize(mean=mean,
                                         std=std)])

        x = compose(fframe) # transform = T.Normalize(mean=mean, std=std)
        
        x = x.detach().numpy()

        # norm_imgT = (torch.from_numpy(x).float())
        
        cv2.imshow("After Normalization", x.transpose((1, 2, 0)))
        x = x.transpose((0, 2, 1))
        x = np.expand_dims(x, axis=0)

        x = torch.from_numpy(x).float()

        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        
        # model prediction
        with torch.no_grad():
            _ = model(x.to(device))
        
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.cpu().detach())
            # print(v.shape)

        # initialize hook outputs
        outputs = []       
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()

        embedding_vectors = embedding_vectors.view(B, C, H * W).to(device)
        dist_list = torch.zeros(size=(H*W, B))
        mean = torch.Tensor(train_outputs[0]).to(device)
        cov_inv = torch.Tensor(train_outputs[1]).to(device)

        delta = (embedding_vectors - mean).permute(2, 0, 1)
        dist_list = (torch.matmul(delta, cov_inv.permute(2, 0, 1)) * delta).sum(2).permute(1, 0)
        dist_list = dist_list.reshape(B, H, W)
        dist_list = dist_list.clamp(0).sqrt().cpu()
        
    
        # upsample
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        
        score_map = np.expand_dims(score_map, axis=0)
        
        # apply gaussian smoothing on the score map   
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)


        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        # Transpose for the fix image
        scores = scores.transpose(0,2,1)

        cv2.imshow('Without Threshold', scores.transpose(1,2,0)) 

        # Getting thresholded mask
        new_mask = get_mask(scores, threshold)

        # Canny edge detection for the mask
        edged = cv2.Canny(np.uint8(new_mask), 0, 255)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        contourPixel = cv2.resize(fframe, (224,224))

        # Draw contours to the output
        cv2.drawContours(contourPixel, contours, -1, (0, 255, 0), 3)
        cv2.imshow('segment', contourPixel)

        threshold = cv2.getTrackbarPos("Threshold", "output") / 100.
        threshold = np.float32(threshold)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

def get_mask(scores, threshold):
    mask = scores[0]
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask *= 255

    return mask


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
