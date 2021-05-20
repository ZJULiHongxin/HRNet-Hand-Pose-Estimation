import argparse
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchvision import transforms
import torch
import torch.backends.cudnn as cudnn

import _init_paths
from fp16_utils.fp16util import network_to_half

from config import cfg
from config import update_config
from utils.utils import get_model_summary

import dataset
from dataset import make_dataloader, make_test_dataloader
from models import pose_hrnet, pose_hrnet_softmax, pose_hrnet_PoseAggr,pose_hrnet_volumetric, multiview_pose_hrnet, CPM
from utils.heatmap_decoding import get_final_preds
import matplotlib.image as gImage
from dataset import build_transforms
import kornia

def main():
    parser = argparse.ArgumentParser(description='Please specify the mode [training/assessment/predicting]')
    parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)
    parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        default=-1,
                        type=int)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--model_path',
                    type=str)
    parser.add_argument('--image_path',
                        type=str)
    args = parser.parse_args()

    model_path = args.model_path

    update_config(cfg, args)

    ngpus_per_node = torch.cuda.device_count()

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.FP16.ENABLED:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if cfg.FP16.STATIC_LOSS_SCALE != 1.0:
        if not cfg.FP16.ENABLED:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    device = 'cpu' if args.gpu == -1 else 'cuda:{}'.format(args.gpu)
    print("Use {}".format(device))

    # model initialization
    model = eval(cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
     # load model state
    print("Loading model:", model_path)
    checkpoint = torch.load(model_path, map_location = 'cpu')

    if 'state_dict' not in checkpoint.keys():
        for key in list(checkpoint.keys()):
            new_key = key.replace("module.", "")
            checkpoint[new_key] = checkpoint.pop(key)
        model.load_state_dict(checkpoint, strict=False)
    else:
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)
        model.load_state_dict(state_dict, strict=False)
        print('Model epoch {}'.format(checkpoint['epoch']))

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    #model = torch.nn.DataParallel(model)
    model.to(device)


    legend_lst = np.array([
    # 0           1               2                  3                4
    'wrist', 'thumb palm', 'thumb near palm', 'thumb near tip', 'thumb tip',
    # 5                    6                 7                8
    'index palm', 'index near palm', 'index near tip', 'index tip',
    # 9                    10                  11               12
    'middle palm', 'middle near palm', 'middle near tip', 'middle tip',
    # 13                  14               15            16
    'ring palm', 'ring near palm', 'ring near tip', 'ring tip',
    # 17                  18               19              20
    'pinky palm', 'pinky near palm', 'pinky near tip', 'pinky tip'])

    legend_dict = OrderedDict(sorted(zip(legend_lst, [i for i in range(21)]), key=lambda x:x[1]))
    
    """
    python inference.py --cfg ../experiments/RHD/RHD_w32_256x256_adam_lr1e-3.yaml --model_path  ../output/RHD_kpt/pose_hrnet/w32_256x256_adam_lr1e-3/model_best.pth.tar --image_path ../test_images/video_rgb.mp4
    python inference.py --cfg ../experiments/MHP/MHP_v1.yaml --model ../output/MHP/MHP_trainable_softmax_v2/MHP_kpt/pose_hrnet_trainable_softmax/MHP_v1/model_best.pth.tar --image_path ../test_images/
    python inference.py --cfg ../experiments/JointTraining/JointTraining_v1.yaml --model_path ../output/JointTraining/JointTraining_v1/model_best.pth.tar --image_path ../test_images/hand.mp4 --gpu cpu
    """
    def predict_one_img(image, show=False, img_path=None):
        trans = build_transforms(cfg, is_train=False)
        temp_joints = [np.ones((21,3))]
        orig_img = image.copy()
        resized_image = cv2.cvtColor(cv2.resize(image, tuple(cfg.MODEL.IMAGE_SIZE)), cv2.COLOR_RGB2BGR)
        I, _ = trans(resized_image, temp_joints)
        I = I.unsqueeze(0).to(device) if args.gpu != 'cpu' else I.unsqueeze(0)

        model.eval()
        with torch.no_grad():
            start_time = time.time()
            output, _ = model(I) # output size: 1 x 21 x 64(H) x 64(W)
            print('Inference time: {:.4f} s'.format(time.time()-start_time))
            kps_pred_np = get_final_preds(output, use_softmax=cfg.MODEL.HEATMAP_SOFTMAX).cpu().numpy().squeeze()
            #kps_pred_np = kornia.spatial_soft_argmax2d(output, normalized_coordinates=False) # 1 x 21 x 2
        #kps_pred_np =  kps_pred_np[0] * np.array([256 / cfg.MODEL.HEATMAP_SIZE[0], 256 / cfg.MODEL.HEATMAP_SIZE[0]])
        

        #kps_pred_np[:,0] += 25
        if True:            
            all_flag = False
            kps_pred_np *=  cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.HEATMAP_SIZE[0]
            heatmap_all = np.zeros(tuple(output.shape[2:]))
            heatmap_lst = []

            fig = plt.figure()           
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(resized_image)

            for kp in range(0,21): 
                heatmap = output[0][kp].cpu().numpy()
                heatmap_lst.append(heatmap)
                heatmap_all += heatmap

            if not all_flag:         
                ax2 = fig.add_subplot(1,2,2)
                heatmap_cat = np.vstack((np.hstack(heatmap_lst[0:7]), np.hstack(heatmap_lst[7:14]), np.hstack(heatmap_lst[14:21])))
                print(heatmap_cat.shape)
                #hm = 255 * output[0][kp] / hms[0][kp].sum()
                ax1.scatter(kps_pred_np[kp][0], kps_pred_np[kp][1], linewidths=10)
                ax2.imshow(heatmap_cat)

                plt.title(kps_pred_np[kp].tolist())
                plt.show()

            if all_flag:
                
                ax2 = fig.add_subplot(1,2,2)
                ax1.imshow(resized_image)
                ax2.imshow(heatmap_all / heatmap_all.max())
                plt.show()
        else:
            kps_pred_np[:,0] *= orig_img.shape[1] / cfg.MODEL.HEATMAP_SIZE[0]
            kps_pred_np[:,1] *= orig_img.shape[0] / cfg.MODEL.HEATMAP_SIZE[0]
            kps_pred_np[:,0] += 0
            fig = plt.figure()
            fig.set_tight_layout(True)
            plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))

            plt.plot([kps_pred_np[0][0], kps_pred_np[legend_dict['thumb palm']][0]], [kps_pred_np[0][1], kps_pred_np[legend_dict['thumb palm']][1]], c='r', marker='.')
            plt.plot(kps_pred_np[1:5,0], kps_pred_np[1:5,1], c='r', marker='.', label='Thumb')
            plt.plot([kps_pred_np[0][0], kps_pred_np[legend_dict['index palm']][0]], [kps_pred_np[0][1], kps_pred_np[legend_dict['index palm']][1]], c='g', marker='.')
            plt.plot(kps_pred_np[5:9,0], kps_pred_np[5:9,1], c='g', marker='.', label='Index')
            plt.plot([kps_pred_np[0][0], kps_pred_np[legend_dict['middle palm']][0]], [kps_pred_np[0][1], kps_pred_np[legend_dict['middle palm']][1]], c='b', marker='.')
            plt.plot(kps_pred_np[9:13,0], kps_pred_np[9:13,1], c='b', marker='.', label='Middle')
            plt.plot([kps_pred_np[0][0], kps_pred_np[legend_dict['ring palm']][0]], [kps_pred_np[0][1], kps_pred_np[legend_dict['ring palm']][1]], c='m', marker='.')
            plt.plot(kps_pred_np[13:17,0], kps_pred_np[13:17,1], c='m', marker='.', label='Ring')
            plt.plot([kps_pred_np[0][0], kps_pred_np[legend_dict['pinky palm']][0]], [kps_pred_np[0][1], kps_pred_np[legend_dict['pinky palm']][1]], c='y', marker='.')
            plt.plot(kps_pred_np[17:21,0], kps_pred_np[17:21,1], c='y', marker='.', label='Pinky')
            plt.title('Prediction')
            if img_path:
                plt.title(img_path)
            plt.axis('off')
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper right", ncol=1, mode="expand", borderaxespad=0.)


        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        buf = fig.canvas.buffer_rgba()

        if show:
            plt.show()
            print(kps_pred_np)


        return np.asarray(buf), kps_pred_np


    if os.path.isdir(args.image_path):
        # a bunch of images
        imgpath_lst = os.listdir(args.image_path)
        for p in imgpath_lst:
            if p.endswith('mp4'):continue
            print(os.path.join(args.image_path, p))
            I = cv2.imread(os.path.join(args.image_path, p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            predict_one_img(I, True, os.path.join(args.image_path, p))
    else:
        try:
            # an image
            I = cv2.imread(args.image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            predict_one_img(I, True, args.image_path)
        except Exception as e:
            print(e)
            # video
            v = cv2.VideoCapture(args.image_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter('pred_results.mp4', fourcc, 10, (640, 480))
            count = 0
            pose2d_pred_lst = []
            while(v.isOpened()):
                ret, frame = v.read()
                count +=1 
                if count < 0: continue
                if ret == False:
                    break
                result, pose2d_pred = predict_one_img(frame, show=False)
                pose2d_pred_lst.append(pose2d_pred)
                print(result.shape)
                videoWriter.write(result[:,:,0:-1])
                if count == 130:
                    break
            np.savetxt('./pose2d_pred.txt', np.concatenate(pose2d_pred_lst, axis=0))
            v.release()
            videoWriter.release()

  

if __name__ == '__main__':
    main()
