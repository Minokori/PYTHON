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

    def _create_carla_instance(self):
        self._generate_start_bat()
        subprocess.Popen(
            ["cmd.exe", "/c", "start", "", r".\start_carla.bat"]
        )

    def _generate_start_bat(self):
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
        assert self._client is not None
        return self._client

    def _wait_for_carla_startup(self):
        start_time = time.time()
        while not self.is_running:
            time.sleep(1.0)
            logging.info(f"等待 Carla Simulator 启动中... 已经等待{(time.time() - start_time):.1f}s")

        self._wait_for_carla_load_world()
        return

    def startup(self):
        if self.is_running:
            logging.info(f"carla has already been running on port {self.config.port}")
            return self
        else:
            self._create_carla_instance()
            self._wait_for_carla_startup()
            return self

    def _wait_for_carla_load_world(self, world_name: str | None = None, map_layers: carla.MapLayer = carla.MapLayer.All):
        self.client.set_timeout(10.0)
        if world_name:
            self.client.load_world_if_different(world_name, map_layers=map_layers)
        start_time = time.time()
        carla_world_info = None
        while carla_world_info is None:
            try:
                carla_world_info = self.client.get_world()
                logging.info(carla_world_info)
            except Exception as e:
                logging.info(f"等待 Carla Simulator 加载世界 `{world_name}`...已经等待{(time.time() - start_time):.1f}s")
                logging.error(e)
                time.sleep(1.0)
        self.client.set_timeout(self.config.waiting_time)
        return

    def change_world(self, world_name: str, map_layers: carla.MapLayer = carla.MapLayer.All):
        self._wait_for_carla_load_world(world_name, map_layers)
        return self
