""" Carla 扩展. 旨在使 Carla 更加易用. """
from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class CarlaConfig():
    executable_path: str = "D:/Carla/0916/CarlaUE4.exe"
    port: int = 2000
    waiting_time: float = 20.0
    use_gpu: bool = True
    low_quality_mode: bool = False

    @property
    def cmd_args(self) -> list[str]:
        args = [f"-world-port={self.port}"]
        if self.use_gpu:
            args.append("-prefernvidia")
        if self.low_quality_mode:
            args.append("-quality-level=Low")
        return args
