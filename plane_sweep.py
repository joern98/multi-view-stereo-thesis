####
# Author: Jörn Eggersglüß
#
# Implementation of the second approach
# Computes image consistency with all images for a family of planes. IR0 is reference camera.
# Outputs a directory inside the input directory containing generated depth map and point clouds
####

import math
import os
import time
from datetime import datetime
from os import path
from typing import Tuple, Sequence

import cv2 as cv
import numpy as np
import argparse
import json
import open3d as o3d

# imported .pyd, ignore error
from plane_sweep_ext import compute_consistency_image

from scipy.interpolate import interp1d

from utility.camera_calibration import load_calibration_from_file
from utility.utilities import save_point_cloud

WINDOW_LEFT_IR_1 = "left IR 1"
WINDOW_LEFT_IR_2 = "left IR 2"
WINDOW_RIGHT_IR_1 = "right IR 1"
WINDOW_RIGHT_IR_2 = "right IR 2"


def load_data(directory: str):
    if directory is None or directory == "":
        raise Exception("Directory not given")
    calibration = load_calibration_from_file(path.join(directory, "Calibration.npy"))
    with open(path.join(directory, "CameraParameters.json")) as f:
        camera_parameters = json.load(f)
    left_ir_1 = cv.imread(path.join(directory, "left_ir_1.png"))
    left_ir_2 = cv.imread(path.join(directory, "left_ir_2.png"))
    right_ir_1 = cv.imread(path.join(directory, "right_ir_1.png"))
    right_ir_2 = cv.imread(path.join(directory, "right_ir_2.png"))
    left_depth = np.load(path.join(directory, "left_depth_raw.npy"))
    right_depth = np.load(path.join(directory, "right_depth_raw.npy"))

    return calibration, camera_parameters, left_ir_1, left_ir_2, right_ir_1, right_ir_2, left_depth, right_depth


def get_camera_matrix_from_dict(intrinsic_dict):
    k = [[intrinsic_dict["fx"], 0, intrinsic_dict["ppx"]],
         [0, intrinsic_dict["fy"], intrinsic_dict["ppy"]],
         [0, 0, 1]]
    return np.asanyarray(k, dtype=np.float32).reshape(3, 3)

def compute_homography(k_rt0: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       k_rt1: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       z: float) -> np.ndarray:
    n = np.asarray([0, 0, -1]).reshape(3, 1)
    K0_I = np.linalg.inv(k_rt0[0])
    K1 = k_rt1[0]
    R = k_rt1[1]
    t = k_rt1[2]
    tn = t @ n.T
    H = K1 @ (R - tn / z) @ K0_I
    return H


def plane_sweep(images: [cv.Mat | np.ndarray | cv.UMat], k_rt: [Tuple[np.ndarray, np.ndarray, np.ndarray]],
                image_size: Sequence[int],
                z_min: float, z_max: float, z_step: float,
                out_directory=None,
                cost_volume=None):
    """
    Perform the basic plane sweeping algorithm

    :param cost_volume:
    :param out_directory:
    :param images: Array of images
    :param k_rt: Array of Tuples (K, R, t)
    :param image_size:
    :param z_min:
    :param z_max:
    :param z_step:
    :return:
    """

    if cost_volume is None:
        n_planes = math.floor((z_max - z_min) / z_step) + 1
        cost_volume = np.zeros((n_planes, image_size[1], image_size[0]), dtype=np.float32)

        # Fill cost volume
        total = 0
        for i in range(n_planes):
            start = time.perf_counter_ns()
            z = z_min + i * z_step
            print(f"Plane at z={z}")
            _L = [images[0]]
            for j in range(1, len(images)):
                # L[0] is not warped, projection would only "scale" the image
                H = compute_homography(k_rt[0], k_rt[j], z)
                projected = cv.warpPerspective(images[j], H, image_size, flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0.5, 0.5, 0.5))
                _L.append(projected)

            # Comment out lines to not show images
            for m in range(len(_L)):
                cv.imshow(f"Camera {m}", _L[m])

            cv.waitKey(1)

            ref = _L[0]
            src = np.asarray(_L[1:])

            compute_consistency_image(ref, src, cost_volume[i], 7)
            dt = time.perf_counter_ns() - start
            total += dt
            print(f"...Took {dt / 1000000} ms")

            # Visualize plane cost
            v = (cost_volume[i] + 1.0) / 2
            cv.imshow("cost_volume", v)
            cv.waitKey(1)

        print(f"Average time per plane: {total / n_planes}")
        save_cost_volume = input(f"Save cost-volume? (Estimated size: {cost_volume.nbytes / 1024} kb) (y/n): ") == 'y'
        if save_cost_volume and out_directory is not None:
            np.save(path.join(out_directory, f"cost_volume_n{n_planes}.npy"), cost_volume)

    else:
        cost_volume = np.load(cost_volume)

    # find depth
    # np.argmax returns the index of max element across axis
    max_idx = np.argmax(cost_volume, axis=0)
    depth = z_min + max_idx * z_step
    m = np.squeeze(np.take_along_axis(cost_volume, max_idx.reshape(1, 720, 1280), axis=0))
    NCC_THRESHOLD = 0.8
    depth = np.where(m > NCC_THRESHOLD, depth, 0)

    return depth


def compute_transforms(calibration_result, camera_parameters) -> [Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    R_01 = np.asarray(camera_parameters["left_stereo_extrinsics"]["r"]).reshape(3, 3)
    t_01 = np.asarray(camera_parameters["left_stereo_extrinsics"]["t"]).reshape(3, 1)

    R_12 = np.asarray(calibration_result.R)
    t_12 = np.asarray(calibration_result.T)

    R_23 = np.asarray(camera_parameters["right_stereo_extrinsics"]["r"]).reshape(3, 3)
    t_23 = np.asarray(camera_parameters["right_stereo_extrinsics"]["t"]).reshape(3, 1)

    R_02 = R_12 @ R_01
    t_02 = R_12 @ t_01 + t_12

    R_03 = R_23 @ R_02
    t_03 = R_23 @ R_12 @ t_01 + R_23 @ t_12 + t_23

    K0 = K1 = get_camera_matrix_from_dict(camera_parameters["left_intrinsics"])
    K2 = K3 = get_camera_matrix_from_dict(camera_parameters["right_intrinsics"])

    R_00 = np.eye(3)
    t_00 = np.zeros((1, 3))

    m = [(K0, R_00, t_00),
         (K1, R_01, t_01),
         (K2, R_02, t_02),
         (K3, R_03, t_03)]
    return m


OUTPUT_DIRECTORY = None


def ensure_output_directory(root_directory):
    global OUTPUT_DIRECTORY
    if OUTPUT_DIRECTORY is None:
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        pathname = path.join(root_directory, f"Output_{timestamp}_plane_sweep")
        os.mkdir(pathname)
        OUTPUT_DIRECTORY = pathname
    return OUTPUT_DIRECTORY


def depth_to_point_cloud(depth: np.ndarray, intrinsic, extrinsic,
                         colormap: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    if colormap is not None:
        depth_image = o3d.geometry.Image(depth)
        # OpenCV is BGR, Open3D expects RGB
        depth_colormap_image = o3d.geometry.Image(colormap)
        depth_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=depth_colormap_image,
                                                                              depth=depth_image,
                                                                              depth_trunc=20,
                                                                              depth_scale=1,
                                                                              convert_rgb_to_intensity=False)
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(depth_rgbd_image, intrinsic, extrinsic)
    else:
        depth_image = o3d.geometry.Image(depth)
        pc = o3d.geometry.PointCloud.create_from_depth_image(depth_image,
                                                             intrinsic,
                                                             extrinsic,
                                                             depth_scale=1,
                                                             depth_trunc=20)
    return pc


def create_pinhole_intrinsic_from_dict(intrinsic_dict, image_size):
    return o3d.camera.PinholeCameraIntrinsic(width=image_size[0],
                                             height=image_size[1],
                                             cx=intrinsic_dict["ppx"],
                                             cy=intrinsic_dict["ppy"],
                                             fx=intrinsic_dict["fx"],
                                             fy=intrinsic_dict["fy"])


def main(args):
    calibration_result, camera_parameters, \
        left_ir_1, left_ir_2, right_ir_1, right_ir_2, \
        left_native_depth, right_native_depth = load_data(args.directory)

    def output_dir():
        return ensure_output_directory(args.directory)

    # greyscale images are still 3-channel, extract the first channel to save memory for plane sweep
    images = [cv.extractChannel(left_ir_1, 0),
              cv.extractChannel(left_ir_2, 0),
              cv.extractChannel(right_ir_1, 0),
              cv.extractChannel(right_ir_2, 0)]
    transforms = compute_transforms(calibration_result, camera_parameters)
    depth = plane_sweep(images, transforms, camera_parameters["image_size"], z_min=0.5, z_max=9.0, z_step=0.02,
                        out_directory=output_dir(), cost_volume=args.cost_volume)

    cv.destroyAllWindows()
    m = interp1d((0, 16), (0, 255), bounds_error=False, fill_value=(0, 255))
    depth_colored = cv.applyColorMap(m(depth).astype(np.uint8), cv.COLORMAP_JET)
    cv.imshow("depth", depth_colored)

    # This is a hack we need for Open3D to be able to create the point cloud: scale depth to mm and convert to uint16
    depth_image = o3d.geometry.Image((depth * 1000).astype(np.uint16))
    intrinsic = create_pinhole_intrinsic_from_dict(camera_parameters["left_intrinsics"],
                                                   camera_parameters["image_size"])
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_image,
                                                                  intrinsic=intrinsic,
                                                                  extrinsic=np.eye(4),
                                                                  depth_scale=1000,
                                                                  depth_trunc=20)

    MOUSE_X, MOUSE_Y = 0, 0

    def on_mouse(event, x, y, flags, user_data):
        nonlocal MOUSE_X, MOUSE_Y
        if event == cv.EVENT_MOUSEMOVE:
            MOUSE_X, MOUSE_Y = x, y
            print(f"depth: {depth[MOUSE_Y, MOUSE_X]} m")

    cv.setMouseCallback("depth", on_mouse)
    print("press 's' to save output")
    key = cv.waitKey()

    if key == ord('s'):
        cv.imwrite(path.join(output_dir(), "Depth_PlaneSweep.png"), depth_colored)
        save_point_cloud(point_cloud, "PointCloud_PlaneSweep", output_directory=output_dir())
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Path to the directory created by capture.py")
    parser.add_argument("--cost_volume", help="Path to cost-volume .npy file")
    args = parser.parse_args()
    main(args)
