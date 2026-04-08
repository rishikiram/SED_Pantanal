from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class PathConfig:
    data_root: str = 'data/birdclef-2026'
    cache_dir: str = 'cache'
    output_dir: str = 'outputs'


@dataclass
class AudioConfig:
    sample_rate: int = 32000
    window_duration: float = 5.0
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 128
    fmin: int = 40
    fmax: int = 15000
    top_db: float = 80.0

    @property
    def samples_per_window(self) -> int:
        return int(self.sample_rate * self.window_duration)

    @property
    def frames_per_window(self) -> int:
        return int(self.samples_per_window / self.hop_length)


@dataclass
class ModelConfig:
    backbone: str = 'efficientnet_b0'
    backbone_pretrained: bool = True
    in_chans: int = 1
    rnn_type: str = 'gru'
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 2
    rnn_bidirectional: bool = True
    rnn_dropout: float = 0.3
    classifier_dropout: float = 0.3
    num_classes: int = 234


@dataclass
class TrainingConfig:
    batch_size: int = 32
    pretrain_clip_epochs: int = 5
    num_epochs: int = 15
    lr: float = 1e-4
    backbone_lr_multiplier: float = 0.1
    weight_decay: float = 1e-2
    loss: str = 'focal_bce'
    focal_gamma: float = 2.0
    grad_clip_norm: float = 5.0
    num_folds: int = 5
    seed: int = 42


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(yaml_path: str) -> Config:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    cfg = Config()
    if 'paths' in raw:
        cfg.paths = PathConfig(**raw['paths'])
    if 'audio' in raw:
        cfg.audio = AudioConfig(**raw['audio'])
    if 'model' in raw:
        cfg.model = ModelConfig(**raw['model'])
    if 'training' in raw:
        cfg.training = TrainingConfig(**raw['training'])
    return cfg
