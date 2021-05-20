import os
import errno
import random
import yaml
import json
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.triangulation_model_utils import multiview
from scipy.optimize import least_squares

def config_to_str(config):
    return yaml.dump(yaml.safe_load(json.dumps(config)))  # fuck yeah

def update_after_resize(intrinsic_matrix, image_shape, new_image_shape):
    height, width = image_shape
    new_height, new_width = new_image_shape

    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    intrinsic_matrix[0, 0] = fx * (new_width / width)
    intrinsic_matrix[1, 1] = fy * (new_height / height)
    intrinsic_matrix[0, 2] = cx * (new_width / width)
    intrinsic_matrix[1, 2] = cy * (new_height / height)

    return intrinsic_matrix

def homogeneous_to_euclidean(points):
    """Converts torch homogeneous points to euclidean
    Args:
        points torch tensor of shape (N, M + 1): N homogeneous points of dimension M
    Returns:
        torch tensor of shape (N, M): euclidean points
    """
    return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)


def euclidean_to_homogeneous(points):
    """Converts torch euclidean points to homogeneous
    Args:
        points torch tensor of shape (N, M): N euclidean points of dimension M
    Returns:
        torch tensor of shape (N, M+1): homogeneous points
    """
    return torch.cat((points, torch.ones(points.shape[0], 1).to(points.device)),-1)


def project3Dto2D(points, projections):
    """Project batch of 3D points to 2D
    Args:
        points torch tensor of shape (B, 3)
        projections torch tensor of shape (B, N, 3, 4)
    Returns:
        torch tensor of shape (B, N, 2)
    """
    points_homogeneous = euclidean_to_homogeneous(points)
    points_homogeneous = points_homogeneous.unsqueeze(1).repeat(1, projections.shape[1], 1)
    points_2d_homogeneous = torch.matmul(projections.reshape(-1,3,4), points_homogeneous.reshape(-1,4,1)).unsqueeze(-1)
    points_2d = homogeneous_to_euclidean(points_2d_homogeneous)
    return points_2d.reshape(projections.shape[0], projections.shape[1], 2)


def DLT_sii_pytorch(points, proj_matricies, number_of_iterations = 2):
    """This module lifts B 2d detections obtained from N viewpoints to 3D using the Direct Linear Transform method.
    It computes the eigenvector associated to the smallest eigenvalue using the Shifted Inverse Iterations algorithm.
    Args:
        proj_matricies torch tensor of shape (B, N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (B, N, 2): sequence of points' coordinates
    Returns:
        point_3d torch tensor of shape (B, 3): triangulated points
    """

    batch_size = proj_matricies.shape[0]
    n_views = proj_matricies.shape[1]

    # assemble linear system
    A = proj_matricies[:,:, 2:3].expand(batch_size, n_views, 2, 4) * points.view(-1, n_views, 2, 1)
    A -= proj_matricies[:, :, :2]
    A = A.view(batch_size, -1, 4)

    AtA = A.permute(0,2,1).matmul(A).float()
    I = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1).to(A.device)
    B =  AtA + 0.001*I
    # initialize normalized random vector
    bk = torch.rand(batch_size, 4, 1).float().to(AtA.device)
    norm_bk = torch.sqrt(bk.permute(0,2,1).matmul(bk))
    bk = bk/norm_bk
    for k in range(number_of_iterations):
        bk, _ = torch.solve(bk, B)
        norm_bk = torch.sqrt(bk.permute(0,2,1).matmul(bk))
        bk = bk/norm_bk

    point_3d_homo = -bk.squeeze(-1)
    point_3d = homogeneous_to_euclidean(point_3d_homo)

    return point_3d

def triangulate_from_multiple_views_svd(proj_matricies, points):
    """This module lifts B 2d detections obtained from N viewpoints to 3D using the Direct Linear Transform method.
    It computes the eigenvector associated to the smallest eigenvalue using Singular Value Decomposition.
    Args:
        proj_matricies torch tensor of shape (B, N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (B, N, 2): sequence of points' coordinates
    Returns:
        point_3d numpy torch tensor of shape (B, 3): triangulated point
    """

    batch_size = proj_matricies.shape[0]
    n_views = proj_matricies.shape[1]

    A = proj_matricies[:,:, 2:3].expand(batch_size, n_views, 2, 4) * points.view(-1, n_views, 2, 1)
    A -= proj_matricies[:, :, :2]

    #_, _, vh = torch.svd(A.view(batch_size, -1, 4))
    _, _, vh = torch.svd(A.view(batch_size, -1, 4))

    point_3d_homo = -vh[:, :, 3]
    point_3d = homogeneous_to_euclidean(point_3d_homo)

    return point_3d

def DLT(pose2d, ext_matrices, int_matrix=None):
    # pose2d: N_views x N_points x 2
    # ext_matrices: N_views x 3 x 4
    # int_matrix: 3 x 3
    # ret_val: recovered 3D poses

    A_lst = [[] for _ in range(pose2d.shape[1])]
    recovered_pose3d_lst = []

    for k in range(pose2d.shape[1]):
        for v in range(pose2d.shape[0]):
            P = int_matrix.dot(ext_matrices[v]) if int_matrix is not None else ext_matrices[v]
            A_lst[k].append(pose2d[v,k,0] * P[2,:] - P[0,:])
            A_lst[k].append(pose2d[v,k,1] * P[2,:] - P[1,:])
        
        A = np.stack(A_lst[k]) # 8(2* N_views) x 4

        eigenvalues,eigenvectors=np.linalg.eig(np.dot(A.T, A))
        pose3d_rec = eigenvectors[:,eigenvalues.argmin()] # (4,)
        pose3d_rec = pose3d_rec[0:3] / pose3d_rec[3]

        recovered_pose3d_lst.append(pose3d_rec)

    return np.stack(recovered_pose3d_lst)

def DLT_pytorch(pose2d, ext_matrices, int_matrix=None):
    # pose2d: N_views x N_points x 2
    # ext_matrices: N_views x 3 x 4
    # int_matrix: 3 x 3
    # ret_val: recovered 3D poses
    ext_matrices = ext_matrices.float()
    A_lst = [[] for _ in range(pose2d.shape[1])]
    recovered_pose3d_lst = []

    for k in range(pose2d.shape[1]):
        for v in range(pose2d.shape[0]):
            P = torch.matmul(int_matrix, ext_matrices[v]) if int_matrix else ext_matrices[v]
            A_lst[k].append(pose2d[v,k,0] * P[2,:] - P[0,:])
            A_lst[k].append(pose2d[v,k,1] * P[2,:] - P[1,:])
        
        A = torch.stack(A_lst[k]) # 8(2* N_views) x 4

        # eigenvalues: 4 x 2 [real, imaginary]; feigenvectors: 4 x 4
        eigenvalues,eigenvectors=torch.eig(torch.matmul(A.T, A), eigenvectors=True)
        #print(eigenvalues,'\n',eigenvectors)
        pose3d_rec = eigenvectors[:,eigenvalues[:,0].argmin()] # (4,)
        #print(pose3d_rec)
        pose3d_rec = pose3d_rec[0:3] / pose3d_rec[3]

        recovered_pose3d_lst.append(pose3d_rec)


    return torch.stack(recovered_pose3d_lst)


def triangulate_ransac(proj_matricies, points, n_iters=10, reprojection_error_epsilon=40, direct_optimization=False):
        # proj_matricies: N_views x 3 x 4
        # points: N_views x 2
        assert len(proj_matricies) == len(points)
        assert len(points) >= 2

        proj_matricies = np.array(proj_matricies)
        points = np.array(points)

        n_views = len(points)

        # determine inliers
        view_set = set(range(n_views))
        inlier_set = set()
        for i in range(n_iters):
            sampled_views = sorted(random.sample(view_set, 2)) # get 2 views randomly

            # recover 3D world coordinates (size: (1,3)) by using DLT
            keypoint_3d_in_base_camera = DLT(points[sampled_views,np.newaxis], proj_matricies[sampled_views])
            # calculate the distance between the groundtruth and the reprojected 2D pose (size: N_views)
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(keypoint_3d_in_base_camera, points, proj_matricies)[0]

            new_inlier_set = set(sampled_views)
            #input(reprojection_error_vector)
            for view in view_set:
                current_reprojection_error = reprojection_error_vector[view]
                if current_reprojection_error < reprojection_error_epsilon:
                    new_inlier_set.add(view)

            if len(new_inlier_set) > len(inlier_set):
                inlier_set = new_inlier_set

        # triangulate using inlier_set
        if len(inlier_set) == 0:
            inlier_set = view_set.copy()

        inlier_list = np.array(sorted(inlier_set))
        inlier_proj_matricies = proj_matricies[inlier_list]
        inlier_points = points[inlier_list]

        keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(inlier_proj_matricies, inlier_points)
        reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
        reprojection_error_mean = np.mean(reprojection_error_vector)
        #print(reprojection_error_vector)
        keypoint_3d_in_base_camera_before_direct_optimization = keypoint_3d_in_base_camera
        reprojection_error_before_direct_optimization = reprojection_error_mean

        # direct reprojection error minimization (using least square to refine the DLT result)
        if direct_optimization:
            def residual_function(x):
                reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([x]), inlier_points, inlier_proj_matricies)[0]
                residuals = reprojection_error_vector
                return residuals

            x_0 = np.array(keypoint_3d_in_base_camera) # initial guess
            res = least_squares(residual_function, x_0, loss='huber', method='trf')

            keypoint_3d_in_base_camera = res.x
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
            reprojection_error_mean = np.mean(reprojection_error_vector)
        #print(inlier_list)
        return keypoint_3d_in_base_camera, inlier_list

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def plot_performance(PCK2d, th2d_lst, mse2d_each_joint):
    legend_lst = [
    # 0           1               2                  3                4
    'wrist', 'thumb palm', 'thumb near palm', 'thumb near tip', 'thumb tip',
    # 5                    6                 7                8
    'index palm', 'index near palm', 'index near tip', 'index tip',
    # 9                    10                  11               12
    'middle palm', 'middle near palm', 'middle near tip', 'middle tip',
    # 13                  14               15            16
    'ring palm', 'ring near palm', 'ring near tip', 'ring tip',
    # 17                  18               19              20
    'pinky palm', 'pinky near palm', 'pinky near tip', 'pinky tip', 'Avg']

    color = ['grey','gold','darkviolet','turquoise','r','g','b', 'c', 'm', 'y',
            'k','darkorange','lightgreen','plum', 'tan',
            'khaki', 'pink', 'skyblue','lawngreen','salmon','coral','maroon']

    # 2D pose mse
    plt.figure(1)
    plt.subplots_adjust(top=0.97, bottom=0.32, left=0.11, right=0.96, hspace=0.2, wspace=0.2)
    X = list(range(0, 2 * (mse2d_each_joint.shape[0] + 1),2))
    Y = np.concatenate((mse2d_each_joint, [mse2d_each_joint.mean()]))
    plt.bar(X, Y, width = 1.5, color = color)
    print('EPE: {:.4f}'.format(mse2d_each_joint.mean()))
    plt.xticks(X, legend_lst, rotation=270)
    plt.xlabel('Key Point')
    plt.ylabel('MSE [px]')
    plt.title('2D pose MSE. Average: {:.4f}'.format(mse2d_each_joint.mean()))
    for x,y in zip(X,Y):
        plt.text(x+0.005,y+0.005,'%.2f' % y, fontsize=6, ha='center',va='bottom')

    # 2/3D pose PCK
    fig = plt.figure(3)
    start, end = 0, 30#len(PCK2d)
    th2d_lst, PCK2d = th2d_lst[start:end], PCK2d[start:end]
    plt.plot(th2d_lst, PCK2d, marker='.')
    plt.xlabel('threshold [px]')
    plt.ylabel('PCK')

    # Area under the curve
    area = (PCK2d[0] + 2 * PCK2d[1:-1].sum() + PCK2d[-1])  * (th2d_lst[1] - th2d_lst[0]) / 2 / (th2d_lst[-1] - th2d_lst[0])
    plt.title('2D PCK AUC over all joints: {:.4f}'.format(area))
    print('2D PCK: {:.4f}'.format(area))
    plt.tight_layout()
    plt.show()

