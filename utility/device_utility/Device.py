import pyrealsense2 as rs


class Device:

    def __init__(self, device, serial):
        self.device: rs.device = device
        self.serial: str = serial
        self.config: rs.config = rs.config()
        self.config.enable_device(self.serial)
        self.pipeline: rs.pipeline = rs.pipeline()
        self.pipeline_profile: rs.pipeline_profile = None
        self.__is_streaming = False

    def is_streaming(self):
        return self.__is_streaming

    def start_stream(self, width, height, fps, streams, record_to_file):
        for s in streams:
            if s is rs.stream.depth:
                self.config.enable_stream(s, width=width, height=height, format=rs.format.z16, framerate=fps)
            elif s is rs.stream.infrared:
                self.config.enable_stream(rs.stream.infrared, 1, width=width, height=height, format=rs.format.y8, framerate=fps)
                self.config.enable_stream(rs.stream.infrared, 2, width=width, height=height, format=rs.format.y8, framerate=fps)
            elif s is rs.stream.color:
                self.config.enable_stream(s, width=width, height=height, format=rs.format.rgb8, framerate=fps)

        if record_to_file is not None:
            self.config.enable_record_to_file(record_to_file)

        self.pipeline_profile = self.pipeline.start(self.config)
        self.__is_streaming = True

    def stop_stream(self):
        self.pipeline.stop()
        self.__is_streaming = False
