from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import torch
import torch.backends.cudnn as cudnn

from load_llff import load_llff_data
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

from config import cfg
from config import update_config
from utils.misc import DLT_pytorch, DLT, triangulate_ransac, DLT_sii_pytorch
from utils.heatmap_decoding import get_final_preds
from models import pose_hrnet, pose_hrnet_softmax,pose_hrnet_volumetric, multiview_pose_hrnet, CPM
from dataset.frei_utils.fh_utils import *
from dataset.HandGraph_utils.vis import *
from dataset import build_transforms
base_dir = r'C:\Users\79233\Desktop\Github\nonrigid_nerf\data\hand'


def vec2rodrig(vec):
    rodrig_mat = [[0 for _ in range(3)] for _ in range(3)]
    rodrig_mat[0][1], rodrig_mat[1][0] = -vec[2], vec[2]
    rodrig_mat[0][2], rodrig_mat[2][0] = vec[1], -vec[1]
    rodrig_mat[1][2], rodrig_mat[2][1] = -vec[0], vec[0]

    if isinstance(vec, list):
        return rodrig_mat
    else:
        return np.array(rodrig_mat)

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
                    default=0,
                    type=int)
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

if cfg.MODEL.SYNC_BN and not args.distributed:
    print('Warning: Sync BatchNorm is only supported in distributed training.')

#model = torch.nn.DataParallel(model)
model.to(device)


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
        # print('Inference time: {:.4f} s'.format(time.time()-start_time))
        kps_pred_np = get_final_preds(output, use_softmax=cfg.MODEL.HEATMAP_SOFTMAX).cpu().numpy().squeeze()
    return kps_pred_np

# images: 114 x H x W x 3
# poses(c2w and intrinsic): 114 x 3 x 5  
images, poses, bds, render_poses, i_test = load_llff_data(
            base_dir,
            factor=3,
            recenter=True
        )

import pickle
fname = './pose2d_pred.txt'
try:
    print('Load')
    with open(fname, 'rb') as f:
        pts = pickle.load(f)
except:
    print('Failed')
    pts = []
    color_lower=(80, 45, 30)
    color_upper=(120, 190, 180)

    images = (images * 255).astype(np.uint8)

    for i in range(images.shape[0]):
        img_HLS = cv2.cvtColor(images[i], cv2.COLOR_BGR2HLS)
        img_HLS_mask = cv2.inRange(img_HLS, color_lower, color_upper) / 255
        img_HLS_mask = np.tile(img_HLS_mask[:,:,np.newaxis], (1,1,3)).astype(img_HLS.dtype)
        img = images[i] * img_HLS_mask
        images[i] = img
        #print(images[i].max(),images[i].min())
        pose2d_pred = predict_one_img(img)
        pose2d_pred[:,0] = images.shape[2] * pose2d_pred[:,0] / 64
        pose2d_pred[:,1] = images.shape[1] * pose2d_pred[:,1] / 64
        pts.append(pose2d_pred)
    pts = np.stack(pts) # n x 21 x 2
    with open(fname, 'wb') as f:
        pickle.dump(pts, f)

hwf = poses[0, :3, -1] # img H, img W, focal length

# intrinsics size: 3 x 3
intrin_m = np.array([[hwf[2], 0, hwf[1]/2],
                      [0, hwf[2], hwf[0]/2],
                      [0,     0,   1]])
print(intrin_m)
poses = poses[:, :3, :4] # c2w size: N_img x 3 x 4

bottom = np.tile([0,0,0,1],(poses.shape[0], 1, 1))
c2w = np.concatenate((poses, bottom), axis=1) # c2w_homo: N_img x 4 x 4
w2c = np.linalg.inv(c2w) # N_img x 4 x 4

print("Loaded llff", images.shape, hwf, base_dir)

# uv = intrin_m * Xc = intrin_m * w2c * Xw
pose3d_pred_lst = []
proj_matrices = intrin_m @ w2c[:,0:-1,:]

for k in range(pts.shape[1]):
    pose3d_pred, _ = triangulate_ransac(
        proj_matrices[:pts.shape[0]], pts[:,k],
        reprojection_error_epsilon=25,
        direct_optimization=False
        )
    pose3d_pred_lst.append(pose3d_pred)
    
pose3d_pred = np.stack(pose3d_pred_lst)
#pose3d_pred = DLT(pts, w2c[:,0:-1,:], intrin_m)
pose3d_pred -= np.mean(pose3d_pred, axis=0, keepdims=True)
pose3d_pred *= np.array([[100,10,100]])

print(pose3d_pred)

"""
compare
"""
def plot_camera(ax):
    ax.plot([5,-5,-5,5,5], [5,5,-5,-5,5],[0,0,0,0,0],'r')
    ax.plot([5,0], [5,0],[0,-15],'r')
    ax.plot([-5,0], [5,0],[0,-15],'r')
    ax.plot([-5,0], [-5,0],[0,-15],'r')
    ax.plot([5,0], [-5,0],[0,-15],'r')

fig = plt.figure(1)
for i in range(1,9):
    ax = fig.add_subplot(2,4,i)
    ax.imshow(images[i])
    plot_hand(ax, pts[i], order='uv', draw_kp=False)

fig = plt.figure(2)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')
ax1.imshow(images[0])
plot_hand(ax1, pts[0], order='uv', draw_kp=True)
draw_3d_skeleton_on_ax(pose3d_pred, ax2)
#plot_camera(ax2)
plt.tight_layout()
plt.show()