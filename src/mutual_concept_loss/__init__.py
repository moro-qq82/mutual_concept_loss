"""Core package exports for the mutual_concept_loss project."""

from .data import (
    DEFAULT_PRIMITIVES,
    GeneratorConfig,
    GroupedBatchSampler,
    PrimitiveContext,
    PrimitiveDefinition,
    PrimitiveInstance,
    SyntheticTaskDataset,
    SyntheticTaskGenerator,
)
from .losses import (
    LossManager,
    LossSchedule,
    SparseAutoencoderLoss,
    SharedSubspaceLoss,
    TaskLoss,
)
from .models import (
    ConvGridDecoder,
    ConvGridEncoder,
    LinearAdapter,
    SharedAutoencoderModel,
    SharedBottleneck,
    SparseAutoencoder,
)
from .training import (
    CheckpointConfig,
    CosineWarmupSchedulerConfig,
    LoggingConfig,
    OptimizerConfig,
    TrainingConfig,
    Trainer,
    build_cosine_warmup_scheduler,
    evaluate_loop,
)
from .utils.seed import set_seed

__all__ = [
    "DEFAULT_PRIMITIVES",
    "GeneratorConfig",
    "GroupedBatchSampler",
    "PrimitiveContext",
    "PrimitiveDefinition",
    "PrimitiveInstance",
    "SyntheticTaskDataset",
    "SyntheticTaskGenerator",
    "LossManager",
    "LossSchedule",
    "SparseAutoencoderLoss",
    "SharedSubspaceLoss",
    "TaskLoss",
    "ConvGridDecoder",
    "ConvGridEncoder",
    "LinearAdapter",
    "SharedAutoencoderModel",
    "SharedBottleneck",
    "SparseAutoencoder",
    "CheckpointConfig",
    "CosineWarmupSchedulerConfig",
    "LoggingConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "Trainer",
    "build_cosine_warmup_scheduler",
    "evaluate_loop",
    "set_seed",
]
