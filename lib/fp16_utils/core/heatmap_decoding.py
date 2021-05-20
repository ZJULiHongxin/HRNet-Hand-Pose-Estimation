import math
import numpy as np
import cv2
import torch
import kornia
import matplotlib.pyplot as plt
#from utils.transforms import transform_preds


def get_max_preds(hms):
    '''
    get predictions from score maps
    params:
    @ hms: numpy.ndarray([batch_size, num_joints, height, width])

    retval:
    @ preds: size B x K (21) x 2 [u→,v↓]
    @ maxvals: size B x K x 1
    '''
    assert isinstance(hms, torch.Tensor), \
        'hms should be torch.Tensor'
    assert hms.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = hms.shape[0]
    num_joints = hms.shape[1]
    width = hms.shape[3]
    heatmaps_reshaped = hms.reshape((batch_size, num_joints, -1)) # flatten a heat map B x 21 x (W*H)
    #idx = torch.argmax(heatmaps_reshaped, 2) # B x 21
    maxvals, idx = torch.max(heatmaps_reshaped, 2) # B x 21

    maxvals = maxvals.reshape((batch_size, num_joints, 1)) # B x 21 x 1
    idx = idx.reshape((batch_size, num_joints, 1)) # B x 21 x 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0]) % width # u axis (col)
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width) # v-axis (row)

    pred_mask = maxvals > 0.0
    pred_mask = pred_mask.repeat(1, 1, 2).float()

    preds *= pred_mask
    return preds, maxvals


def taylor(hm, coord):
    """
    approximate the true maximum point from the coarse max point using 2nd order Taylor expansion

    params:
    @ hm: a heat map of size H x W
    @ coord: the uv-position of the maximum element. size: 2 [u→,v↓]
    """
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px, py = int(coord[0]), int(coord[1])

    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx  = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
            + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2][px] - 2 * hm[py][px] + hm[py-2][px])

        derivative = torch.tensor([[dx],[dy]]) # (2,1)
        hessian = torch.tensor([[dxx,dxy],[dxy,dyy]])

        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = torch.inverse(hessian)
            offset = torch.mm(-hessianinv, derivative)
            offset = torch.squeeze(offset.T, dim=0)
            coord += offset.cuda()

    return coord


def gaussian_blur(hms, kernel):
    """
    Heatmap distribution modulation

    the heatmaps predicted by a human pose estimation model do not exhibit good-shaped Gaussian structure
    compared to the training heatmap data. we propose to exploit a Gaussian kernel K with the same variation
    as the training data to smooth out the effects of multiple peaks in the heatmap while keeping the
    original maximum activation location.

    params:
    @ hms: numpy.ndarray([batch_size, 21(num_joints), height, width])
    @ kernel: int
    """
    border = (kernel - 1) // 2
    batch_size = hms.shape[0]
    num_joints = hms.shape[1]
    height = hms.shape[2]
    width = hms.shape[3]
    sigma = (kernel-1)//3

    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = torch.max(hms[i,j])
            dr = torch.zeros((height + 2 * border, width + 2 * border)) # zero-padding
            dr[border: -border, border: -border] = hms[i,j].clone()
            dr = kornia.gaussian_blur2d(dr.view(1,1,dr.shape[0],dr.shape[1]), (kernel, kernel), (sigma,sigma))

            hms[i,j] = dr[0,0,border: -border, border: -border].clone()
            hms[i,j] *= origin_max / torch.max(hms[i,j])
    return hms


def get_final_preds(hms, config):
    """
    params:
    @ hms: torch.Tensor([batch_size, num_joints, height, width])
    retval:
    @ preds: size: B x 21 x 2
    """  
    # post-processing
    # hms_np = hms.cpu().detach().numpy()[0][0]
    # f1 = plt.figure(3)
    # ax1 = f1.add_subplot(1,3,1)
    # ax2 = f1.add_subplot(1,3,2)
    # ax3 = f1.add_subplot(1,3,3)

    # ax1.imshow(hms_np)
    # ax1.set_title('before Gaussian blur')

    #hms = gaussian_blur(hms, config.MODEL.SIGMA*3+1)

    # hms_np = hms.cpu().detach().numpy()[0][0]
    # ax2.imshow(hms_np)
    # ax2.set_title('Gau blur')
    # sum_of_each_map = torch.sum(hms.view(hms.shape[0], hms.shape[1], -1),dim=2)
    # for b in range(hms.shape[0]):
    #     for k in range(hms.shape[1]):
    #         hms[b][k] /= sum_of_each_map[b][k]
    #hms = kornia.geometry.subpix.spatial_softmax2d(hms, temperature=25.70796)
    # hms_np = hms.cpu().detach().numpy()[0][0]
    # ax3.imshow(hms_np)
    # ax3.set_title('after soft max')
    # plt.show()

    preds = kornia.geometry.subpix.spatial_expectation2d(hms, normalized_coordinates=False)
    return preds

# import math
# import numpy as np
# import cv2
# #from utils.transforms import transform_preds


# def get_max_preds(hms):
#     '''
#     get predictions from score maps
#     params:
#     @ hms: numpy.ndarray([batch_size, num_joints, height, width])

#     retval:
#     @ preds: size B x K (21) x 2 [u→,v↓]
#     @ maxvals: size B x K x 1
#     '''
#     assert isinstance(hms, np.ndarray), \
#         'hms should be numpy.ndarray'
#     assert hms.ndim == 4, 'batch_images should be 4-ndim'

#     batch_size = hms.shape[0]
#     num_joints = hms.shape[1]
#     width = hms.shape[3]
#     heatmaps_reshaped = hms.reshape((batch_size, num_joints, -1)) # flatten a heat map
#     idx = np.argmax(heatmaps_reshaped, 2)
#     maxvals = np.amax(heatmaps_reshaped, 2)

#     maxvals = maxvals.reshape((batch_size, num_joints, 1))
#     idx = idx.reshape((batch_size, num_joints, 1))

#     preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

#     preds[:, :, 0] = (preds[:, :, 0]) % width # u axis (col)
#     preds[:, :, 1] = np.floor((preds[:, :, 1]) / width) # v-axis (row)

#     pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
#     pred_mask = pred_mask.astype(np.float32)

#     preds *= pred_mask
#     return preds, maxvals


# def taylor(hm, coord):
#     """
#     approximate the true maximum point from the coarse max point using 2nd order Taylor expansion

#     params:
#     @ hm: a heat map of size H x W
#     @ coord: the uv-position of the maximum element. size 1 x 2 [u→,v↓]
#     """
#     heatmap_height = hm.shape[0]
#     heatmap_width = hm.shape[1]
#     px = int(coord[0])
#     py = int(coord[1])
#     if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
#         dx  = 0.5 * (hm[py][px+1] - hm[py][px-1])
#         dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
#         dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
#         dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
#             + hm[py-1][px-1])
#         dyy = 0.25 * (hm[py+2][px] - 2 * hm[py][px] + hm[py-2][px])
#         derivative = np.matrix([[dx],[dy]])
#         hessian = np.matrix([[dxx,dxy],[dxy,dyy]])
#         if dxx * dyy - dxy ** 2 != 0:
#             hessianinv = hessian.I
#             offset = -hessianinv * derivative
#             offset = np.squeeze(np.array(offset.T), axis=0)
#             coord += offset
#     return coord


# def gaussian_blur(hms, kernel):
#     """
#     Heatmap distribution modulation

#     the heatmaps predicted by a human pose estimation model do not exhibit good-shaped Gaussian structure
#     compared to the training heatmap data. we propose to exploit a Gaussian kernel K with the same variation
#     as the training data to smooth out the effects of multiple peaks in the heatmap while keeping the
#     original maximum activation location.

#     params:
#     @ hms: numpy.ndarray([batch_size, num_joints, height, width])
#     @ kernel: int
#     """
#     border = (kernel - 1) // 2
#     batch_size = hms.shape[0]
#     num_joints = hms.shape[1]
#     height = hms.shape[2]
#     width = hms.shape[3]
#     for i in range(batch_size):
#         for j in range(num_joints):
#             origin_max = np.max(hms[i,j])
#             dr = np.zeros((height + 2 * border, width + 2 * border)) # zero-padding
#             dr[border: -border, border: -border] = hms[i,j].clone()
#             dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
#             hms[i,j] = dr[border: -border, border: -border].clone()
#             hms[i,j] *= origin_max / np.max(hms[i,j])
#     return hms


# def get_final_preds(hms, config):
#     """
#     params:
#     @ hms: numpy.ndarray([batch_size, num_joints, height, width])
#     retval:
#     @ preds: size: B x 21 x 2
#     """
#     coords, maxvals = get_max_preds(hms) # coords: size B x K (21) x 2 [u,v]; maxvals: size B x K x 1

#     # post-processing
#     hms = gaussian_blur(hms, config.MODEL.SIGMA*3+1)

#     # In order to reduce the approximation difficulty, we use
#     # logarithm to transform the original exponential form G to a
#     # quadratic form P to facilitate inference
#     hms = np.maximum(hms, 1e-10)  #np.maximum：(X, Y, out=None)#X 与 Y 逐位比较取其大者；最少接收两个参数
#     hms = np.log(hms)
#     for b in range(coords.shape[0]): # batch index
#         for k in range(coords.shape[1]): # key point index
#             coords[b,k] = taylor(hms[b][k], coords[b][k])

#     preds = coords.clone()

#     return preds, maxvals
