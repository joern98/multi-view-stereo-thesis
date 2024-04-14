import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, fields
from datetime import datetime
from json import JSONEncoder
from typing import Tuple, Sequence

import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from .device_utility.DeviceManager import DeviceManager
from .device_utility.DevicePair import DevicePair
from .device_utility.utils import set_sensor_option, get_stereo_extrinsic, get_sensor_option

NUM_PATTERNS_REQUIRED = 7
# https://docs.opencv.org/4.x/d9/d5d/classcv_1_1TermCriteria.html, (TYPE, iterations, epsilon)
TERM_CRITERIA = (cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_EPS, 30, 0.001)
WINDOW_IMAGE_LEFT = "left ir"
WINDOW_IMAGE_RIGHT = "right ir"


# cv.UMat is np.ndarray internally
@dataclass()
class CalibrationResult:
    # return values: retval (rms), cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, perViewErrors
    # tuple[float, UMat, UMat, UMat, UMat, UMat, UMat, UMat, UMat, UMat
    rms: float
    camera_matrix_left: np.ndarray
    coeffs_left: np.ndarray
    camera_matrix_right: np.ndarray
    coeffs_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    per_view_errors: np.ndarray
    R_14: np.ndarray | None  # optional 4x4 transformation matrix from outer left to outer right
    image_size: Tuple[int, int]


@dataclass()
class CameraParameters:
    left_rs_intrinsics: rs.intrinsics
    left_camera_matrix: np.ndarray
    left_dist_coeffs: np.array
    right_rs_intrinsics: rs.intrinsics
    right_camera_matrix: np.ndarray
    right_dist_coeffs: np.array
    image_size: Tuple[int, int]

    # left camera stereo extrinsic [left IR(1) -> right IR(2)]
    left_stereo_extrinsics: rs.extrinsics

    # right camera stereo extrinsic [left IR(1) -> right IR(2)]
    right_stereo_extrinsics: rs.extrinsics


@dataclass()
class RectificationResult:
    left_map_x: np.ndarray
    left_map_y: np.ndarray
    right_map_x: np.ndarray
    right_map_y: np.ndarray
    R_left: np.ndarray
    R_right: np.ndarray
    P_left: np.ndarray
    P_right: np.ndarray
    Q: np.ndarray
    ROI_left: Sequence[int]
    ROI_right: Sequence[int]


def run_camera_calibration(device_pair: DevicePair, rows=7, columns=10, size=24) -> Tuple[
    CalibrationResult, RectificationResult]:
    cv.namedWindow(WINDOW_IMAGE_RIGHT)
    cv.namedWindow(WINDOW_IMAGE_LEFT)
    # exposure unit is microseconds -> [0, 166000] 166ms
    cv.createTrackbar("exposure", WINDOW_IMAGE_LEFT, 0, 166000, lambda v: change_exposure_time(v, device_pair))
    cv.setTrackbarPos("exposure", WINDOW_IMAGE_LEFT,
                      int(get_sensor_option(device_pair.left.device.first_depth_sensor(), rs.option.exposure)))

    # we only need IR streams
    device_pair.start(width=1280, height=720, fps=30, streams=(rs.stream.infrared,))

    left_ir_index = 2
    right_is_index = 1

    camera_parameters = collect_camera_parameters(device_pair, left_ir_index, right_is_index)

    object_points, image_points_left, image_points_right = find_chessboard_corners(device_pair, left_ir_index,
                                                                                   right_is_index,
                                                                                   pattern_dimensions=(rows, columns),
                                                                                   pattern_size=(size, size))

    calibration_result = stereo_calibrate(camera_parameters, object_points, image_points_left,
                                          image_points_right)

    # transform calibration
    calibration_result.R_14 = transpose_inner_to_outer_stereo(camera_parameters, calibration_result)
    rectification_result = stereo_rectify(camera_parameters.image_size, calibration_result)

    device_pair.stop()
    cv.destroyAllWindows()

    # TODO do not return calibration result as is, only return parameters for outer cameras

    save_prompt_result = input("save calibration? (y/n): ")
    if save_prompt_result == 'y':
        write_calibration_to_file(calibration_result)

    return calibration_result, rectification_result


def collect_camera_parameters(device_pair: DevicePair, left_ir_index=1, right_ir_index=2) -> CameraParameters:
    # inner IR cameras
    left_intrinsic: rs.intrinsics = device_pair.left.pipeline_profile.get_stream(rs.stream.infrared, left_ir_index) \
        .as_video_stream_profile().get_intrinsics()
    right_intrinsic: rs.intrinsics = device_pair.right.pipeline_profile.get_stream(rs.stream.infrared, right_ir_index) \
        .as_video_stream_profile().get_intrinsics()

    left_camera_matrix = rs_intrinsics_to_camera_matrix(left_intrinsic)
    right_camera_matrix = rs_intrinsics_to_camera_matrix(right_intrinsic)

    left_coefficients = np.array(left_intrinsic.coeffs).astype(np.float32)
    right_coefficients = np.array(right_intrinsic.coeffs).astype(np.float32)

    left_stereo_extrinsic = get_stereo_extrinsic(device_pair.left.pipeline_profile)
    right_stereo_extrinsic = get_stereo_extrinsic(device_pair.right.pipeline_profile)

    camera_params = CameraParameters(left_rs_intrinsics=left_intrinsic,
                                     left_camera_matrix=left_camera_matrix,
                                     left_dist_coeffs=left_coefficients,
                                     left_stereo_extrinsics=left_stereo_extrinsic,
                                     right_rs_intrinsics=right_intrinsic,
                                     right_camera_matrix=right_camera_matrix,
                                     right_dist_coeffs=right_coefficients,
                                     right_stereo_extrinsics=right_stereo_extrinsic,
                                     image_size=(left_intrinsic.width, left_intrinsic.height))
    return camera_params


def change_exposure_time(value, device_pair: DevicePair):
    depth_sensor_left: rs.depth_sensor = device_pair.left.device.first_depth_sensor()
    depth_sensor_right: rs.depth_sensor = device_pair.right.device.first_depth_sensor()
    set_sensor_option(depth_sensor_left, rs.option.exposure, value)
    set_sensor_option(depth_sensor_right, rs.option.exposure, value)


# https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html
# pattern dimensions/size (rows, columns), size in mm (integer)
def find_chessboard_corners(device_pair: DevicePair, left_ir=1, right_ir=2,
                            pattern_dimensions=(5, 7), pattern_size=(34, 34)):
    # turn of ir pattern emitters
    depth_sensor_left: rs.depth_sensor = device_pair.left.device.first_depth_sensor()
    depth_sensor_right: rs.depth_sensor = device_pair.right.device.first_depth_sensor()
    set_sensor_option(depth_sensor_left, rs.option.emitter_enabled, False)
    set_sensor_option(depth_sensor_right, rs.option.emitter_enabled, False)

    # set exposure to below 33ms to allow for 30fps streaming
    set_sensor_option(depth_sensor_left, rs.option.exposure, 30000)
    set_sensor_option(depth_sensor_right, rs.option.exposure, 30000)

    print(f"Find chessboard corners: {pattern_dimensions}, {pattern_size} mm")

    # Initialize array to hold the 3D-object coordinates of the inner chessboard corners
    # Reference: https://docs.opencv.org/4.9.0/dc/dbb/tutorial_py_calibration.html
    objp = np.zeros((pattern_dimensions[0] * pattern_dimensions[1], 3), np.float32)

    # create coordinate pairs for the corners and write them to the array, leaving the z-coordinate at 0
    objp[:, :2] = np.mgrid[0:pattern_dimensions[0] * pattern_size[0]:pattern_size[0],
                  0:pattern_dimensions[1] * pattern_size[1]:pattern_size[1]].T.reshape(-1, 2)  # pattern size 34mm, 5x7

    objp /= 1000  # convert mm to m here to avoid issues with floating point arithmetic in np.mgrid

    object_points = []
    image_points_left = []
    image_points_right = []

    cooldown = False

    def reset_cooldown():
        nonlocal cooldown
        cooldown = False

    # get chessboard corners until required number of valid correspondences has been found
    while np.size(object_points, 0) < NUM_PATTERNS_REQUIRED:
        frame_left, frame_right = device_pair.wait_for_frames()

        # check frame timestamps in ms
        ts_l = frame_left.get_timestamp()
        ts_r = frame_right.get_timestamp()
        d_ts = abs(ts_l - ts_r)

        # inner cameras
        ir_left = frame_left.get_infrared_frame(left_ir)
        ir_right = frame_right.get_infrared_frame(right_ir)
        image_left = np.array(ir_left.get_data())
        image_right = np.array(ir_right.get_data())

        d_ts_too_high = d_ts > 30.0
        if d_ts_too_high:
            print(f"d_ts too high: {d_ts}")
        if not cooldown:
            ret_l, corners_left = cv.findChessboardCornersSB(image_left, pattern_dimensions,
                                                             flags=cv.CALIB_CB_ACCURACY | cv.CALIB_CB_EXHAUSTIVE)
            ret_r, corners_right = cv.findChessboardCornersSB(image_right, pattern_dimensions,
                                                              flags=cv.CALIB_CB_ACCURACY | cv.CALIB_CB_EXHAUSTIVE)
            cv.drawChessboardCorners(image_left, pattern_dimensions, corners_left, ret_l)
            cv.drawChessboardCorners(image_right, pattern_dimensions, corners_right, ret_r)
            if ret_l and ret_r:
                object_points.append(objp)  # corresponding object points

                image_points_left.append(corners_left)
                image_points_right.append(corners_right)

                print(f"{np.size(object_points, 0)} of {NUM_PATTERNS_REQUIRED}")

                # set cooldown period
                cooldown = True
                threading.Timer(2, reset_cooldown).start()

        cv.imshow(WINDOW_IMAGE_LEFT, image_left)
        cv.imshow(WINDOW_IMAGE_RIGHT, image_right)

        if cv.waitKey(1) == 27:  # ESCAPE
            print(f"chessboard corner process aborted, found {np.size(object_points, 0)} sets of correspondences")
            break

    # turn emitters back on
    set_sensor_option(depth_sensor_left, rs.option.emitter_enabled, True)
    set_sensor_option(depth_sensor_right, rs.option.emitter_enabled, True)

    return object_points, image_points_left, image_points_right


def rs_intrinsics_to_camera_matrix(intrinsics: rs.intrinsics) -> np.ndarray:
    m = np.zeros((3, 3), np.float32)
    m[0, 0] = intrinsics.fx
    m[1, 1] = intrinsics.fy
    m[0, 2] = intrinsics.ppx
    m[1, 2] = intrinsics.ppy
    m[2, 2] = 1
    return m


# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga9d2539c1ebcda647487a616bdf0fc716
def stereo_calibrate(camera_params: CameraParameters, object_points, image_points_left,
                     image_points_right):
    per_view_errors = np.zeros(np.size(object_points, 0), np.float32)
    # initialize t and r with initial guess, inner cameras
    r = np.eye(3, 3, dtype=np.float64)
    t = np.zeros((3, 1), np.float64)
    t[0, 0] = -0.095
    result = cv.stereoCalibrate(objectPoints=object_points,
                                imagePoints1=image_points_left,
                                imagePoints2=image_points_right,
                                cameraMatrix1=camera_params.left_camera_matrix,
                                distCoeffs1=camera_params.left_dist_coeffs,
                                cameraMatrix2=camera_params.right_camera_matrix,
                                distCoeffs2=camera_params.right_dist_coeffs,
                                imageSize=camera_params.image_size,
                                R=r,
                                T=t,
                                perViewErrors=per_view_errors,
                                flags=cv.CALIB_FIX_INTRINSIC | cv.CALIB_USE_EXTRINSIC_GUESS)

    # set R_14 if its outer pair
    calibration_result = CalibrationResult(*result, R_14=None, image_size=camera_params.image_size)
    print(f"stereo calibration rms: {calibration_result.rms}")
    print(f"stereo calibration T: {calibration_result.T}")
    return calibration_result


def transpose_inner_to_outer_stereo(camera_params: CameraParameters, calib: CalibrationResult):
    """
    Return the 4x4 transformation matrix R_14=(R|t) in homogenous coordinates
    :param camera_params:
    :param calib: Calibration result from calibrating the inner cameras of the device pair
    :return:
    """
    # R14 = R34*R23*R12
    # R12 and R34 are camera extrinsic parameters
    # R23 is calibration result
    R_12 = np.eye(4, dtype=np.float32)
    R_23 = np.eye(4, dtype=np.float32)
    R_34 = np.eye(4, dtype=np.float32)

    # rs.extrinsics.rotation is column-major 3x3 matrix -> transpose to row major for compatibility with openCV
    R_12[:3, :3] = np.asarray(camera_params.left_stereo_extrinsics.rotation).reshape(3, 3).T
    R_12[:3, 3:4] = np.asarray(camera_params.left_stereo_extrinsics.translation).reshape(3, 1)

    R_34[:3, :3] = np.asarray(camera_params.right_stereo_extrinsics.rotation).reshape(3, 3).T
    R_34[:3, 3:4] = np.asarray(camera_params.right_stereo_extrinsics.translation).reshape(3, 1)

    # calib.R is already row-major as it was created by openCV
    R_23[:3, :3] = calib.R
    R_23[:3, 3:4] = calib.T

    # @ is shorthand for np.matmul(a, b)
    R_14 = R_34 @ R_23 @ R_12
    return R_14


# Reference: https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/#stereo-rectification
# https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4
def stereo_rectify(image_size: Tuple[int, int], calib: CalibrationResult):
    # transform inner camera calibration to outer camera calibration transform
    if calib.R_14 is not None:
        R = calib.R_14[:3, :3].astype(np.float64)
        T = calib.R_14[:3, 3:4].astype(np.float64)
    else:
        R = calib.R
        T = calib.T

    # All the matrices must have the same data type in function 'cvRodrigues2' -> convert to float64
    # https://answers.opencv.org/question/3441/strange-stereorectify-error-with-rotation-matrix/ -> double precision
    # camera params are the same as calib result
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(cameraMatrix1=calib.camera_matrix_left,
                                                     distCoeffs1=calib.coeffs_left,
                                                     cameraMatrix2=calib.camera_matrix_right,
                                                     distCoeffs2=calib.coeffs_right,
                                                     imageSize=(1280, 720),  # fix hardcoded size
                                                     R=R,
                                                     T=T,
                                                     alpha=1)
    # map_1 -> map_x, map_2 -> map_y
    left_map_1, left_map_2 = cv.initUndistortRectifyMap(cameraMatrix=calib.camera_matrix_left,
                                                        distCoeffs=calib.coeffs_left,
                                                        R=R1,
                                                        newCameraMatrix=P1,
                                                        size=image_size,
                                                        m1type=cv.CV_32FC1)
    right_map_1, right_map_2 = cv.initUndistortRectifyMap(cameraMatrix=calib.camera_matrix_right,
                                                          distCoeffs=calib.coeffs_right,
                                                          R=R2,
                                                          newCameraMatrix=P2,
                                                          size=image_size,
                                                          m1type=cv.CV_32FC1)

    rectification_result = RectificationResult(left_map_1,
                                               left_map_2,
                                               right_map_1,
                                               right_map_2,
                                               R1, R2, P1, P2, Q,
                                               roi1, roi2)
    return rectification_result


class CalibrationResultEncoder(JSONEncoder):

    def __serialize_calibration_result(self, obj: CalibrationResult):
        o = {
            "rms": obj.rms if isinstance(obj.rms, float) else float(obj.rms),
            "camera_matrix_left": obj.camera_matrix_left.tolist(),
            "distortion_coefficients_left": obj.coeffs_left.ravel().tolist(),
            "camera_matrix_right": obj.camera_matrix_right.tolist(),
            "distortion_coefficients_right": obj.coeffs_right.ravel().tolist(),
            "R": obj.R.tolist(),
            "T": obj.T.ravel().tolist(),
            "E": obj.E.tolist(),
            "F": obj.F.tolist(),
            "per_view_errors": obj.per_view_errors.tolist(),
            "R_14": "Direct outer calibration, see R and T" if obj.R_14 is None else obj.R_14.tolist(),
            "image_size": obj.image_size if isinstance(obj.image_size, Tuple) else obj.image_size.tolist()
        }
        return o

    def default(self, obj):
        if isinstance(obj, CalibrationResult):
            return self.__serialize_calibration_result(obj)
        return json.JSONEncoder.default(self, obj)


def write_calibration_to_file(calibration_result: CalibrationResult,
                              file_basename=f"Calibration_{datetime.now().strftime('%y%m%d_%H%M%S')}"):
    with open(file_basename + ".json", "x") as f:
        json.dump(calibration_result, f, cls=CalibrationResultEncoder, indent=2)
        print(f"Written human-readable calibration data to file: {file_basename + '.json'}")

    with open(file_basename + ".npy", "xb") as f:
        # order matters, save all fields of Calibration result
        for field in fields(CalibrationResult):
            attr = getattr(calibration_result, field.name)
            # if R_14 is none, save it as 0, since we cannot save None directly without using pickle
            np.save(f, attr if attr is not None else [0])
        print(f"Written binary calibration data to file: {file_basename + '.npy'}")


def load_calibration_from_file(filename: str) -> CalibrationResult:
    with open(filename, "rb") as f:
        # this is hacky, iterate through the fields and load the data in that order from the .npy file
        calibration_result = CalibrationResult(None, None, None, None, None, None, None, None, None, None, None, None)
        for field in fields(CalibrationResult):
            value = np.load(f)
            # see above for R_14
            if np.array_equal(value, [0]):
                value = None
            setattr(calibration_result, field.name, value)

    print(f"Calibration loaded from {filename}:\n{calibration_result}")
    return calibration_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Camera calibration")
    parser.add_argument("--rows", help="number of rows in the chessboard pattern", default=7, type=int)
    parser.add_argument("--columns", help="number of columns in the chessboard pattern", default=10, type=int)
    parser.add_argument("--size", help="chessboard square size in mm", default=24, type=int)
    args = parser.parse_args()
    ctx = rs.context()
    device_manager = DeviceManager(ctx)
    try:
        left_serial, right_serial = DeviceManager.serial_selection()
    except Exception as e:
        print("Serial selection failed: \n", e)
        sys.exit(0)

    device_pair = device_manager.create_device_pair(left_serial, right_serial)

    depth_sensor_left: rs.depth_sensor = device_pair.left.device.first_depth_sensor()
    depth_sensor_right: rs.depth_sensor = device_pair.right.device.first_depth_sensor()
    set_sensor_option(depth_sensor_left, rs.option.enable_auto_exposure, 0)
    set_sensor_option(depth_sensor_right, rs.option.enable_auto_exposure, 0)

    calibration_result, rectification_result = run_camera_calibration(device_pair, args.rows, args.columns, args.size)

    print(calibration_result)

    # we don't need to print the maps, they are not human-readable anyway
    rectification_result.left_map_x = None
    rectification_result.left_map_y = None
    rectification_result.right_map_x = None
    rectification_result.right_map_y = None
    print(rectification_result)
