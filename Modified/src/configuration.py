from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class NetworkConfiguration:
    debug_level: int
    write_prediction_file: bool

    dt: float
    sequence_lenght: int
    cell_points: int

    model_name: str
    model_id: int

    model_restart: bool
    restart_path: str

    initial_epochs: int

    enable_model_save: bool
    model_save_path: str

    hidden_dimensions: List[int]

    batch_size: int

    total_epochs: int
    validation_split_ratio: float
    validate_from_test_dataset: bool

    learning_rate: float
    decay_epochs: int
    decay_rate: float
    decay_stair_case: bool

    test_dataset_path: str
    train_dataset_path: str

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> NetworkConfiguration:
        return cls(**dictionary)
