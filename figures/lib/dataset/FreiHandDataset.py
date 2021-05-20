import os
import shutil
import time
import logging
from collections import defaultdict, OrderedDict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from .standard_legends import std_legend_lst, idx_Frei
from dataset.frei_utils.fh_utils import load_db_annotation, projectPoints, db_size, plot_hand

logger = logging.getLogger(__name__)

class FreiHandDataset(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where dataset is located to.
        dataset (string): Dataset name(train2017, val2017, test2017).
        data_format(string): Data format for reading('jpg', 'zip')
        transform (callable, optional): A function/transform that  takes in an opencv image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, set_name, data_format, transform=None,
                 target_transform=None):
        self.name = 'Frei'

        self.data_dir = root # FreiHAND
        
        if set_name in ['train', 'training']:
            self.sample_lst = range(0,110000)
        elif set_name in ['eval', 'valid', 'val', 'evaluation', 'validation']:
            self.sample_lst = range(110000,130240)
        
        self.transform = transform
        self.target_transform = target_transform
        self.db_data_anno = list(load_db_annotation(base_path=root, set_name='training'))

        self.img_dir = os.listdir(os.path.join(self.data_dir, 'training', 'rgb'))
        #self.images = sorted(self.img_dir)
        #self.masks = sorted(self.mask_dir)

        self.mask_dir = os.listdir(os.path.join(self.data_dir, 'training', 'mask'))
        
        self.bg_dir = ''
        self.joint_label_list = std_legend_lst

        logger.info(self.__repr__())
        logger.info('=> classes: {}'.format(self.joint_label_list))

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img_path = os.path.join(self.data_dir, 'training', 'rgb', '%08d.jpg' % idx)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        msk = cv2.imread(os.path.join(self.data_dir, 'mask', '%08d.jpg' % idx))

        K, mano, xyz = self.db_data_anno[idx % 32560]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]

        uv = projectPoints(xyz, K) # numpy array of size 21 x 2

        # # show
        # fig = plt.figure()
        # ax1 = fig.add_subplot(121)
        # ax2 = fig.add_subplot(122)
        # ax1.imshow(img)
        # plot_hand(ax1, uv, order='uv')
        # plot_hand(ax2, uv, order='uv')
        # ax1.axis('off')
        # ax2.axis('off')
        # plt.show()

        joints = np.concatenate((uv, np.ones((21,1))), axis=1)

        if self.transform is not None:
            img, joints_list = self.transform(img,[joints])
            return img, joints_list[0], img_path

        if self.target_transform is not None:
            joints = self.target_transform(joints)
        
        return img, msk, joints
        

    def __len__(self):
        return len(self.sample_lst)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            p = keypoints[keypoints[:, 2] > 0][:, :2].mean(axis=0)
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]

        return tmp

    def evaluate(self, config, preds, scores, output_dir,
                 *args, **kwargs):
        '''
        Perform evaluation on RHD keypoint task
        :param config: config dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args: 
        :param kwargs: 
        :return: 
        '''
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % self.data_dir)

        # preds is a list of: batchsize x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)
        kpts = defaultdict(list)
        for idx, _kpts in enumerate(preds):
            img_id = self.ids[idx]
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            for idx_kpt, kpt in enumerate(_kpts):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self.processKeypoints(kpt)
                # if self.with_center:

                kpts[int(file_name[-16:-4])].append(
                    {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        'tags': kpt[:, 3],
                        'image': int(file_name[-16:-4]),
                        'area': area
                    }
                )

        # rescoring and oks nms
        oks_nmsed_kpts = []
        # image x person x (keypoints)
        for img in kpts.keys():
            # person x (keypoints)
            img_kpts = kpts[img]
            # person x (keypoints)
            # do not use nms, keep all detections
            keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file
        )

        if 'test' not in self.dataset:
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder
            )
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0

    def generate_videos(self):
        import random
        output_dir = os.path.join(self.data_dir, 'videos')

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        sample_num = self.__len__()
        video_num = 1
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_num = 100
        tol = 0.5
        video_count = 0
        for i in range(sample_num):
            if video_count == video_num:
                break
            # starting pose
            img_path = os.path.join(self.data_dir, 'rgb', self.images[i])
            mask_path = os.path.join(self.data_dir, 'mask', self.masks[i])
        
            imgk = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            imgk = cv2.cvtColor(imgk, cv2.COLOR_BGR2RGB)

            maskk = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
            Kk, mano, xyzk = self.db_data_anno[i]
            Kk, _, xyzk = [np.array(x) for x in [Kk, mano, xyzk]]

            self.maskout(imgk, maskk)

            # end pose
            end_idx = random.randint(0, sample_num)
            img_path = os.path.join(self.data_dir, 'rgb', self.images[end_idx])
            mask_path = os.path.join(self.data_dir, 'mask', self.masks[end_idx])
        
            imgN = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            maskN = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)

            KN, mano, xyzN = self.db_data_anno[i]
            KN, _, xyzN = [np.array(x) for x in [KN, mano, xyzN]]
            
            self.maskout(imgN, maskN)

            video_frames = [imgk]

            count = 1
            flag = 1

            id_visitd = [i,end_idx]
            while True:
                temp_xyzk = xyzk - count /(frame_num-1) * (xyzk - xyzN)
                print(count)
                for j in range(sample_num):
                    if j in id_visitd:
                        continue
                    id_visitd.append(j)
                    Kk, mano, xyzk = self.db_data_anno[j]
                    _, _, xyzk = [np.array(x) for x in [Kk, mano, xyzk]]

                    dist = np.sum(np.linalg.norm(xyzk - temp_xyzk, axis=1))

                    if dist < tol:
                        img_path = os.path.join(self.data_dir, 'rgb', self.images[j])
                        mask_path = os.path.join(self.data_dir, 'mask', self.masks[j])
                        imgk = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                        maskk = cv2.cvtColor(cv2.imread(mask_path, -1), cv2.COLOR_BGR2GRAY)

                        self.maskout(imgk, maskk)
                        # cv2.imshow('1',imgk)
                        # cv2.waitKey(0)
                        video_frames.append(imgk)
                        count += 1
                        break
                else:
                    flag = 0

                if flag == 0:
                    del video_frames
                    break
                if count == frame_num - 2:
                    break
            
            if flag:
                video_path = os.path.join(output_dir, 'VIDEO_{:06d}.avi'.format(video_count))
                output_movie = cv2.VideoWriter(video_path, fourcc, 5, (224,224))
                for frame in video_frames:
                    output_movie.write(frame)
                output_movie.write(imgN)
                print(video_path,'finished')
                output_movie.release()
                video_count += 1

    def maskout(self, img, mask):
        for ch in range(3):
            img[:,:,ch] = np.bitwise_and(img[:,:,ch], mask)
        
    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []
        num_joints = 17

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))]
            )
            key_points = np.zeros(
                (_key_points.shape[0], num_joints * 3),
                dtype=np.float
            )

            for ipt in range(num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            for k in range(len(img_kpts)):
                kpt = key_points[k].reshape((num_joints, 3))
                left_top = np.amin(kpt, axis=0)
                right_bottom = np.amax(kpt, axis=0)

                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]

                cat_results.append({
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'bbox': list([left_top[0], left_top[1], w, h])
                })

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
            # info_str.append(coco_eval.stats[ind])

        return info_str


