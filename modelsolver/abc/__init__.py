"""各种 `modelsolver` 组件的接口.

---

+ `config` 配置项. 内置的配置项包括
    + 训练超参数配置 `HyperParameterConfig`
    + 数据源配置 `DataConfig`
    + (RL) 经验回放池配置 `ReplayBufferConfig`
    + (RL) 智能体超参数配置 `AgentHyperParameterConfig`

+ `data` 数据处理. 内置的数据处理器包括
    + 数据加载器接口 `IDataLoader`
    + 数据集接口 `IDataset`
    + 数据预处理器接口 `IDataProcesser`
    + (RL) 经验回放池接口 `IReplayBuffer`

+ `environment` 环境. 内置的环境接口包括
    + (RL) 环境接口 `IEnvironment`
"""
