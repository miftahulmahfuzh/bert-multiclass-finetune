from typing import Any, Optional

from pydantic_settings import BaseSettings


class TrainingConfig(BaseSettings):
    lr: float
    epoch: int


class Config(BaseSettings):
    data: str
    data_kwargs: dict[str, Any] = {}
    pretrained_model: str
    num_validation_examples: Optional[int] = None
    num_test_examples: Optional[int] = None
    max_train_examples: int = 0
    text_column: str = "text"
    training: TrainingConfig
    dataset_processor: str | None = None
    ort: bool
    from_flax: bool = False

    batch_size: int

    max_input_length: int = 256
    max_target_length: int = 256

    source: str
    target: str

    prefix: str = ""

    def params(self):
        return {
            "data": self.data,
            "data_kwargs": self.data_kwargs,
            "pretrained_model": self.pretrained_model,
            "num_validation_examples": self.num_validation_examples,
            "num_test_examples": self.num_test_examples,
            "max_train_examples": self.max_train_examples,
            "training_lr": self.training.lr,
            "training_epoch": self.training.epoch,
            "batch_size": self.batch_size,
            "max_input_length": self.max_input_length,
            "max_target_length": self.max_target_length,
            "source": self.source,
            "target": self.target,
            "input_prefix": self.prefix,
            "ort": int(self.ort),
            "from_flax": int(self.from_flax),
        }
