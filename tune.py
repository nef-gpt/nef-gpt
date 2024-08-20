from lightning import Trainer
from ray.train.lightning import (
    RayDDPStrategy,
    RayFSDPStrategy,
    RayDeepSpeedStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from data.loader import ShapeNetData, ShapeNetDataConfig
from models.nano_gpt import NanoGPTConfig
from nef_gpt import LRConfig, NefGPT, OptimizerConfig


def train_func(config, working_dir=None):

    # dataconfig is not tuned, so we will use the default values
    # except for the batch_size

    # due to a quirk in tune, file loading fails
    # we use the TUNE_ORIG_WORKING_DIR environment variable to get the original working directory

    import os

    vq_path = os.path.join(
        working_dir,
        "weights/rq_search_results/shapenet_retrained_learnable_rq_model_dim_1_vocab_127_batch_size_16_threshold_ema_dead_code_0_kmean_iters_1_num_quantizers_1_use_init_False.pth",
    )
    print(f"working dir {working_dir}")
    mlps_path = os.path.join(working_dir, "datasets/shapenet_nef_2/pretrained")
    data_config = ShapeNetDataConfig(
        mlps_folder=mlps_path,
        vector_quantize_pth=vq_path,
        batch_size=config["batch_size"],
    )

    dm = ShapeNetData(data_config=data_config)

    model = NefGPT(
        config=NanoGPTConfig.from_preset(
            "extra_small", dm.get_tokenizer(), dm.get_sequence_length()
        ),
        optimizer_config=OptimizerConfig(
            learning_rate=config["learning_rate"], weight_decay=config["weight_decay"]
        ),
        lr_config=LRConfig(),
    )

    logger = WandbLogger(project="nef-gpt")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        logger=logger,
        devices="auto",
        accelerator="auto",
        # strategy=RayFSDPStrategy(),
        # strategy=RayDDPStrategy(),
        # strategy=RayDeepSpeedStrategy(),
        callbacks=[RayTrainReportCallback(), lr_monitor],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
        # other trainer args
        log_every_n_steps=2,
        precision="bf16-mixed",
        default_root_dir="./.lightning/nef-gpt",
        deterministic=True,
        overfit_batches=24,
    )
    # trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)


def main():
    search_space = {
        "batch_size": tune.choice([2, 4]),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        # maybe also some architecture hyperparameters
    }

    num_epochs = 5
    num_samples = 20

    scaling_config = ScalingConfig(num_workers=1)
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="train/loss",
            checkpoint_score_order="min",
        ),
    )
    import os

    # get normal working directory
    working_dir = os.getcwd()

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        tune.with_parameters(train_func, working_dir=working_dir),
        scaling_config=scaling_config,
        run_config=run_config,
    )

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="train/loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


if __name__ == "__main__":
    main()
