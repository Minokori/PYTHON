import logging
import subprocess
import time

import carla
import win32com.client

from carla_extension import CarlaConfig


class CarlaSimulatorManager():
    """管理与 `Carla` 连接的类

    >>> manager = CarlaSimulatorManager()
    carla_client = manager.startup()
                          .change_world("Town03")
                          .client
    """

    def __init__(self, config: CarlaConfig = CarlaConfig()) -> None:
        self.config = config
        self._client: carla.Client | None = None
        self._camera_delta_transform = carla.Transform(carla.Location(z=30), carla.Rotation(yaw=-90))
        self.ego: carla.Vehicle | None = None

    def _create_carla_instance(self):
        """通过命令行执行 carla"""
        self._generate_start_bat()
        subprocess.Popen(
            ["cmd.exe", "/c", "start", "", r".\start_carla.bat"]
        )

    def _generate_start_bat(self):
        """生成一个运行 carla 的.bat脚本"""
        with open("start_carla.bat", "w") as bat_file:
            bat_file.write(f"@echo off\n")
            bat_file.write(f"start {self.config.executable_path} {" ".join(self.config.cmd_args)}\n")
            bat_file.write(f"exit\n")
        logging.info("生成启动脚本: start_carla.bat")

    @property
    def is_running(self) -> bool:
        wmi = win32com.client.GetObject('winmgmts:')
        processCodeCov = wmi.ExecQuery(f"select * from Win32_Process where name=\"CarlaUE4.exe\"")
        return len(processCodeCov) > 0

    @property
    def client(self) -> carla.Client:
        if self._client is None:
            self._client = carla.Client("localhost", self.config.port)
            self._client.set_timeout(self.config.waiting_time)
        assert self._client is not None
        return self._client

    def _wait_for_carla_startup(self, max_wait: float = 180.0):
        """等待 Carla 进程启动并且能接受连接。加入短超时和总体超时，避免死循环。"""
        start_time = time.time()
        # 等待进程存在
        while not self.is_running:
            if time.time() - start_time > max_wait:
                raise TimeoutError(f"等待 Carla 进程启动超时（{max_wait}s）")
            time.sleep(1.0)
            logging.info(f"等待 Carla Simulator 启动中... 已经等待{(time.time() - start_time):.1f}s")

        # 等待服务器接受连接：使用短超时快速探测
        connect_start = time.time()
        server_version = None
        while server_version is None:
            try:
                server_version = self.client.get_server_version()
                break
            except Exception as e:
                if time.time() - connect_start > max_wait:
                    raise TimeoutError(f"等待 Carla Server 可连接超时（{max_wait}s）") from e
                logging.info(f"等待 Carla Server 接受连接... 已经等待{(time.time() - connect_start):.1f}s")
        logging.info(f"成功连接到 Carla 进程, 服务器 Carla 版本: {server_version}")

        # 恢复默认超时并加载默认世界
        self.client.set_timeout(self.config.waiting_time)

        return

    def startup(self):
        """启动 carla"""
        if self.is_running:
            logging.info(f"carla has already been running on port {self.config.port}")
            return self
        else:
            self._create_carla_instance()
            self._wait_for_carla_startup()
            # 缓存一些常用的对象
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            self.camera = self.world.get_spectator()
            return self

    def _wait_for_carla_load_world(self, world_name: str | None = None, map_layers: carla.MapLayer = carla.MapLayer.All, max_wait: float = 120.0):
        """等待 world 可用并（可选）加载指定世界。加入总体超时与更稳健的重试。"""
        if not world_name:
            return
        start_time = time.time()

        self.client.load_world_if_different(world_name, map_layers=map_layers)

        world = None
        while world is None:
            try:
                world = self.client.get_world()

            except Exception as e:
                logging.info(f"等待 Carla 加载世界 `{world_name}`... 已经等待{(time.time() - start_time):.1f}s")
        return

    def change_world(self, world_name: str, map_layers: carla.MapLayer = carla.MapLayer.All):
        self._wait_for_carla_load_world(world_name, map_layers)
        return self

    def camera_follow(self,follow_ego_heading=True, yaw=0.0):
        assert self.ego is not None, "Ego vehicle is not initialized."
        camera_transform = self.ego.get_transform()
        camera_transform.location += self._camera_delta_transform.location
        camera_transform.rotation.pitch = -90.0
        camera_transform.rotation.roll = 0.0
        if not follow_ego_heading:
            camera_transform.rotation.yaw = -90.0
        self.camera.set_transform(camera_transform)
