#! Default trainer for the nef-gpt model

from data.loader import ShapeNetData, ShapeNetDataConfig
from models.nef_gpt import NefGPT, NanoGPTConfig, OptimizerConfig, LRConfig
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger


def create_trainer(vq_path=default_vq_path, mlps_path=default_mlps_path):
    pass


def create_trainer(
    data_config: ShapeNetDataConfig,
    model_config: NanoGPTConfig,
    optimizer_config: OptimizerConfig,
    lr_config: LRConfig,
):
    data = ShapeNetData(data_config=data_config)

    model = NefGPT(
        config=NanoGPTConfig.from_preset(
            "extra_small", data.get_tokenizer(), data.get_sequence_length()
        ),
        optimizer_config=OptimizerConfig(),
        lr_config=LRConfig(),
    )

    logger = WandbLogger(project="nef-gpt")

    # This adds additional logging on wandb for the model (gradients and topology)
    logger.watch(model)

    trainer = trainer(
        max_epochs=5,
        # used to limit the number of batches for testing and initial overfitting
        limit_train_batches=8,
        limit_val_batches=2,
        # logging stuff
        log_every_n_steps=2,
        logger=logger,
        profiler="simple",
        # performance stuff
        precision="bf16-mixed",
        default_root_dir="./.lightning/nef-gpt",
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
