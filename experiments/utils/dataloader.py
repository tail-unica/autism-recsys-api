"""
Utilities for data loading in experiments.
Load the data from Neo4j database into appropriate structures for hopwise.
"""
import os
from functools import lru_cache
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

import neo4j
import polars as pl

CONFIG_PATH = os.path.join(os.pardir, "config")  # experiments/ -> config/

@lru_cache(maxsize=1)
def get_cfg():
    # Initialize Hydra only once
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(config_path=CONFIG_PATH, version_base=None):
        return compose(config_name="dataset")

def get_index(session: neo4j.Session) -> list:
    # return every relation in the neo4j database as a triple (source, relation, target)
    query = """MATCH (a)-[r]->(b)
               RETURN DISTINCT elementId(a) AS source_id, labels(a) AS source_type, type(r) AS relation, elementId(b) AS target_id, labels(b) AS target_type"""
    result = session.run(query)
    index = []
    for record in result:
        index.append({
            record["source_id"]: record["source_type"][0],
            "relation": record["relation"],
            record["target_id"]: record["target_type"][0],
        })
    return index

def load_data(session: neo4j.Session):
    """Load data from Neo4j and return as a Polars DataFrame."""
    cfg = get_cfg()
    loader = DataLoader(cfg)
    return loader(session)

class BaseConfig:
    def __init__(self, config: dict) -> None:
        self._self = self.__class__.__name__
        self._file = False
        self._type = None
        self._origin = None
        self._from = None
        self._yield = None
        self._link = None
        self._target = None
        self._children = {}

        for key, value in config.items():
            switch = {
                "__self__": lambda v: setattr(self, "_self", v),
                "__file__": lambda v: setattr(self, "_file", v),
                "__type__": lambda v: setattr(self, "_type", v),
                "__origin__": lambda v: setattr(self, "_origin", v),
                "__from__": lambda v: setattr(self, "_from", v),
                "__yield__": lambda v: setattr(self, "_yield", v),
                "__link__": lambda v: setattr(self, "_link", v),
                "__target__": lambda v: setattr(self, "_target", v),
            }
            if key in switch:
                switch[key](value)
            else:
                self._children[key] = value

    def __repr__(self) -> str:
        return f"{self._self} -> FileConfig(type={self._type}, from={self._from}, origin={self._origin}, yield={self._yield}, children={list(self._children.keys())})"
        
    def set_id(self, id: str):
        self._self = id
        return self

class FileConfig(BaseConfig):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        for child_name, child_config in self._children.items():
            self._children[child_name] = ColumnConfig(child_config).set_id(child_name)

    def __call__(self, session: neo4j.Session):
        """Create a new polars DataFrame based on the configuration."""
        # TODO: Implement data loading logic here
        return None

class ColumnConfig(BaseConfig):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

class DataLoader:
    def __init__(self, config) -> None:
        self.config = config
        self._network = {}
        for file_type, file_config in config.items():
            self._network[file_type] = FileConfig(file_config).set_id(file_type)

    def __call__(self, session: neo4j.Session):
        data_frames = {}
        for file_type, file_loader in self._network.items():
            print(f"Loading file type: {file_type}")
            data_frames[file_type] = file_loader(session)
        return data_frames