import pyrealsense2 as rs


def set_sensor_option(sensor: rs.sensor, option: rs.option, value) -> bool:
    if sensor.supports(option):
        sensor.set_option(option, value)
        return True
    else:
        print(f"{sensor}, sn: {sensor.get_info(rs.camera_info.serial_number)} does not support option {option}!")
        return False


def get_sensor_option(sensor: rs.sensor, option: rs.option):
    if sensor.supports(option):
        value = sensor.get_option(option)
        return value
    else:
        print(f"{sensor}, sn: {sensor.get_info(rs.camera_info.serial_number)} does not support option {option}!")
        return None


def get_stereo_extrinsic(profile: rs.pipeline_profile) -> rs.extrinsics:
    """
    Return the extrinsic parameters of the two IR streams of one device,
    i.e. the position and rotation of the right IR camera relative to the left IR camera
    :param profile: Pipeline Profile with both IR streams enabled
    :return: extrinsic parameters from left IR (IR 1) to right IR (IR 2)
    """
    # https://dev.intelrealsense.com/docs/api-how-to#get-disparity-baseline
    ir0_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir1_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    e = ir0_profile.get_extrinsics_to(ir1_profile)
    return e
