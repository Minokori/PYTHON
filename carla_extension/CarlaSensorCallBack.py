"""管理 Carla 的传感器"""


import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import carla
import numpy as np
from dataclasses_json import dataclass_json


def Vector3DtoTuple(vector: carla.Vector3D) -> tuple[float, float, float]:
    return (vector.x, vector.y, vector.z)


@dataclass_json()
@dataclass()
class ImuSensorData:
    gyroscope: tuple[float, float, float]
    compass: float
    accelerometer: tuple[float, float, float]

    @classmethod
    def fromImuData(cls, data: carla.IMUMeasurement):
        gyr = Vector3DtoTuple(data.gyroscope)
        cps = np.degrees(data.compass)
        acc = Vector3DtoTuple(data.accelerometer)
        return cls(gyr, cps, acc)

    def __post_init__(self):
        # TODO 角度没有进行有意义的预处理
        # TODO 保留小数位数
        # TODO 设置阶段阈值?
        pass


@dataclass_json()
@dataclass()
class CollisionSensorData:
    collision_object: str
    impulse: float
    frame: int
    location: tuple[float, float, float]

    @classmethod
    def fromCollisionData(cls, data: carla.CollisionEvent):
        obj_type = data.other_actor.type_id or None

        if obj_type is not None:
            imp_vec = data.normal_impulse.length()
            frm = data.frame
            loc = Vector3DtoTuple(data.transform.location)
            return cls(obj_type, imp_vec, frm, loc)
        else:
            return None


@dataclass_json()
@dataclass()
class GnssSensorData:
    longitude: float
    latitude: float
    location: tuple[float, float, float]

    @classmethod
    def fromGnssData(cls, data: carla.GnssMeasurement):
        lon = data.longitude
        lat = data.latitude
        loc = Vector3DtoTuple(data.transform.location)
        return cls(lon, lat, loc)


@dataclass_json()
@dataclass()
class LaneInvasionSensorData:
    lanes: Sequence[tuple[str, str]]
    location: tuple[float, float, float]

    @classmethod
    def fromLaneInvasionData(cls, data: carla.LaneInvasionEvent):
        # lane_type = set(x.type for x in data.crossed_lane_markings)
        lanes = [(i.type, i.lane_change.name) for i in data.crossed_lane_markings] or None
        if lanes is not None:
            loc = Vector3DtoTuple(data.transform.location)
            return cls(lanes, loc) # type: ignore
        else:
            return None


@dataclass_json()
@dataclass()
class RadarData:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    v_x: np.ndarray
    v_y: np.ndarray

    @classmethod
    def fromRadarSensorData(cls, data: carla.RadarMeasurement):
        points_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4))

        z = points_data[:, 2] * np.cos(points_data[:, 1])
        x = points_data[:, 2] * np.sin(points_data[:, 1]) * np.cos(points_data[:, 0])
        y = points_data[:, 2] * np.sin(points_data[:, 1]) * np.sin(points_data[:, 0])
        v_x = points_data[:, 3] * np.sin(points_data[:, 1]) * np.cos(points_data[:, 0])
        v_y = points_data[:, 3] * np.sin(points_data[:, 1]) * np.sin(points_data[:, 0])
        return cls(x, y, z, v_x, v_y)


class SENSORTYPE(Enum):
    IMUSENSOR = 0,
    COLLISIONSENSOR = 1,
    LANEINVASIONSENSOR = 2,
    GNSSSENSOR = 3,
    RADARSENSOR = 4,


class SensorManager():
    """传感器数据管理"""

    def __init__(self) -> None:
        self.imu_info: ImuSensorData | None = None
        self.gnss_info: GnssSensorData | None = None
        self.collision_info: CollisionSensorData | None = None
        self.lane_info: LaneInvasionSensorData | None = None
        self.radar_info: RadarData | None = None

    def __call__(self, sensor: SENSORTYPE, sensor_data: carla.SensorData):
        if sensor == SENSORTYPE.IMUSENSOR:
            assert isinstance(sensor_data, carla.IMUMeasurement)
            self.__imu_sensor_callback(sensor_data)
        elif sensor == SENSORTYPE.COLLISIONSENSOR:
            assert isinstance(sensor_data, carla.CollisionEvent)
            self.__collision_sensor_callback(sensor_data)
        elif sensor == SENSORTYPE.LANEINVASIONSENSOR:
            assert isinstance(sensor_data, carla.LaneInvasionEvent)
            self.__lane_invasion_sensor_callback(sensor_data)
        elif sensor == SENSORTYPE.GNSSSENSOR:
            assert isinstance(sensor_data, carla.GnssMeasurement)
            self.__gnss_sensor_callback(sensor_data)
        elif sensor == SENSORTYPE.RADARSENSOR:
            assert isinstance(sensor_data, carla.RadarMeasurement)
            self.__radar_sensor_callback(sensor_data)

    def __imu_sensor_callback(self, sensor_data: carla.IMUMeasurement):

        self.imu_info = ImuSensorData.fromImuData(sensor_data)

    def __collision_sensor_callback(self, sensor_data: carla.CollisionEvent):
        self.collision_info = CollisionSensorData.fromCollisionData(sensor_data)
        if self.collision_info is not None:
            logging.info(f"Collision with {self.collision_info.collision_object}")
        else:
            logging.info("no Collision happen.")

    def __lane_invasion_sensor_callback(self, sensor_data: carla.LaneInvasionEvent):
        self.lane_info = LaneInvasionSensorData.fromLaneInvasionData(sensor_data)
        if self.lane_info is not None:
            logging.info(f"Crossed line {self.lane_info.lanes}")
        else:
            logging.info("no lane crossed.")

    def __gnss_sensor_callback(self, sensor_data: carla.GnssMeasurement):
        self.gnss_info = GnssSensorData.fromGnssData(sensor_data)

    def __radar_sensor_callback(self, sensor_data: carla.RadarMeasurement):
        self.radar_info = RadarData.fromRadarSensorData(sensor_data)
