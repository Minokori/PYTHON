from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ReplayBufferConfig:
    capacity: int
    state_dim: int
    action_dim: int
