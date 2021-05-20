import cv2
import glob, os
import numpy as np
import re
import fnmatch
import pickle
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

import torch

import _init_paths
from dataset.HandGraph_utils.utils import *
from dataset.HandGraph_utils.vis import *
from dataset.standard_legends import idx_MHP
import utils
from utils.misc import DLT, DLT_pytorch, DLT_sii_pytorch

baseDir =  r'E:\Hand_Datasets\MHP'
pathToDataset = os.path.join(baseDir, "annotated_frames")

def saveAnnotation(jointCamPath, positions):
	fOut = open(jointCamPath, 'w')
	fOut.write("F4_KNU1_A " + str(positions[0][0]) + " " + str(positions[0][1]) + "\n")
	fOut.write("F4_KNU1_B " + str(positions[1][0]) + " " + str(positions[1][1]) + "\n")
	fOut.write("F4_KNU2_A " + str(positions[2][0]) + " " + str(positions[2][1]) + "\n")
	fOut.write("F4_KNU3_A " + str(positions[3][0]) + " " + str(positions[3][1]) + "\n")

	fOut.write("F3_KNU1_A " + str(positions[4][0]) + " " + str(positions[4][1]) + "\n")
	fOut.write("F3_KNU1_B " + str(positions[5][0]) + " " + str(positions[5][1]) + "\n")
	fOut.write("F3_KNU2_A " + str(positions[6][0]) + " " + str(positions[6][1]) + "\n")
	fOut.write("F3_KNU3_A " + str(positions[7][0]) + " " + str(positions[7][1]) + "\n")

	fOut.write("F1_KNU1_A " + str(positions[8][0]) + " " + str(positions[8][1]) + "\n")
	fOut.write("F1_KNU1_B " + str(positions[9][0]) + " " + str(positions[9][1]) + "\n")
	fOut.write("F1_KNU2_A " + str(positions[10][0]) + " " + str(positions[10][1]) + "\n")
	fOut.write("F1_KNU3_A " + str(positions[11][0]) + " " + str(positions[11][1]) + "\n")

	fOut.write("F2_KNU1_A " + str(positions[12][0]) + " " + str(positions[12][1]) + "\n")
	fOut.write("F2_KNU1_B " + str(positions[13][0]) + " " + str(positions[13][1]) + "\n")
	fOut.write("F2_KNU2_A " + str(positions[14][0]) + " " + str(positions[14][1]) + "\n")
	fOut.write("F2_KNU3_A " + str(positions[15][0]) + " " + str(positions[15][1]) + "\n")

	fOut.write("TH_KNU1_A " + str(positions[16][0]) + " " + str(positions[16][1]) + "\n")
	fOut.write("TH_KNU1_B " + str(positions[17][0]) + " " + str(positions[17][1]) + "\n")
	fOut.write("TH_KNU2_A " + str(positions[18][0]) + " " + str(positions[18][1]) + "\n")
	fOut.write("TH_KNU3_A " + str(positions[19][0]) + " " + str(positions[19][1]) + "\n")
	fOut.write("PALM_POSITION " + str(positions[20][0]) + " " + str(positions[20][1]) + "\n")
	fOut.close()

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def recursive_glob(rootdir='.', pattern='*'):
	matches = []
	for root, dirnames, filenames in os.walk(rootdir):
	  for filename in fnmatch.filter(filenames, pattern):
		  matches.append(os.path.join(root, filename))

	return matches

def readAnnotation3D(file):
	f = open(file, "r")
	an = []
	for l in f:
		l = l.split()
		an.append((float(l[1]),float(l[2]), float(l[3])))

	return np.array(an, dtype=float)

def getCameraMatrix():
	Fx = 614.878
	Fy = 615.479
	Cx = 313.219
	Cy = 231.288
	cameraMatrix = np.array([[Fx, 0, Cx],
					[0, Fy, Cy],
					[0, 0, 1]])
	return cameraMatrix

def getDistCoeffs():
	return np.array([0.092701, -0.175877, -0.0035687, -0.00302299, 0])

# iterate sequences
cameraMatrix = getCameraMatrix()
distCoeffs = getDistCoeffs()

for i in range(1,22):
    # read the color frames
    path = pathToDataset+"/data_"+str(i)+"/"
    colorFrames = recursive_glob(path, "*_webcam_[0-9]*")
    colorFrames = natural_sort(colorFrames)
    print("There are",len(colorFrames),"color frames on the sequence data_"+str(i))
    # read the calibrations for each camera
    print("Loading calibration for calibrations/data_"+str(i))
    # they are py27 pickles so we load as bytes and use the latin1 encoding to open them in py36
    c = [[None for i in range(2)] for j in range(4)]
    c[0][0] = pickle.load(open(baseDir+"/calibrations/data_"+str(i)+"/webcam_1/rvec.pkl","rb"), encoding='latin1')
    c[0][1] = pickle.load(open(baseDir+"/calibrations/data_"+str(i)+"/webcam_1/tvec.pkl","rb"), encoding='latin1')
    c[1][0] = pickle.load(open(baseDir+"/calibrations/data_"+str(i)+"/webcam_2/rvec.pkl","rb"), encoding='latin1')
    c[1][1] = pickle.load(open(baseDir+"/calibrations/data_"+str(i)+"/webcam_2/tvec.pkl","rb"), encoding='latin1')
    c[2][0] = pickle.load(open(baseDir+"/calibrations/data_"+str(i)+"/webcam_3/rvec.pkl","rb"), encoding='latin1')
    c[2][1] = pickle.load(open(baseDir+"/calibrations/data_"+str(i)+"/webcam_3/tvec.pkl","rb"), encoding='latin1')
    c[3][0] = pickle.load(open(baseDir+"/calibrations/data_"+str(i)+"/webcam_4/rvec.pkl","rb"), encoding='latin1')
    c[3][1] = pickle.load(open(baseDir+"/calibrations/data_"+str(i)+"/webcam_4/tvec.pkl","rb"), encoding='latin1')

    for j in range(len(colorFrames)):
        img_path = colorFrames[j]
        #print(img_path)
        img_name, datadir = os.path.basename(img_path), os.path.basename(os.path.dirname(img_path))
        img_idx = img_name.split('_')[0]
        jointPath = os.path.join(baseDir, 'annotations', datadir, img_idx+"_joints.txt")
        #print(jointPath)
        pose3d = readAnnotation3D(jointPath)[0:21][idx_MHP] # size 21 x 3 the last point is the palm

        # read 4 cameras
        pose2d_all_views = []
        proj_matrices_lst = []
        for cam_idx in range(4):
            rvec, tvec = c[cam_idx][0], c[cam_idx][1]
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            rigidMatrix = np.concatenate((rotation_matrix, tvec.reshape((3,1))), axis=1) # 3 x 4
            proj_matrices_lst.append(cameraMatrix @ rigidMatrix)
            # project 3D points (world coord) onto the image plane
            p2d_cam = np.dot(rotation_matrix, np.transpose(pose3d, (1,0))) + tvec.reshape((3,1))
            zeros = np.array([0.,0.,0.])
            pose2d, _ = cv2.projectPoints(p2d_cam, zeros, zeros, cameraMatrix, distCoeffs*0) # 21 x 1 x 2
            pose2d_all_views.append(np.squeeze(pose2d))
        
        proj_matrices = torch.from_numpy(np.stack(proj_matrices_lst)).unsqueeze(0) # 1 x 4 x 3 x 4
        pose2d_all_views = torch.from_numpy(np.stack(pose2d_all_views)).unsqueeze(0) # 1 x 4 x 21 x 2

        pose3d_rec_sii = torch.cat([DLT_sii_pytorch(proj_matrices, pose2d_all_views[:,:,k], 1000) for k in range(pose3d.shape[0])], dim=0) # 21 x 3
        pose3d_rec_DLT = DLT_pytorch(pose2d_all_views[0], proj_matrices[0])

        for k in range(pose3d.shape[0]):
            print(pose3d[k], '\t', pose3d_rec_sii[k], '\t', pose3d_rec_DLT[k])
        
        input()





            