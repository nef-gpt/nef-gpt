from lightning import Trainer
import ray
from ray.train.lightning import (
    RayDDPStrategy,
    RayFSDPStrategy,
    RayDeepSpeedStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune.search.bayesopt import BayesOptSearch
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import torch

from data.loader import ShapeNetData, ShapeNetDataConfig
from models.nano_gpt import NanoGPTConfig
from nef_gpt import LRConfig, NefGPT, OptimizerConfig

tune_settings = {
    "max_epochs": 50,
    "init_tune_steps": 10,
    "total_tune_steps": 60, 
}


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
        batch_size=8, # config["batch_size"],
    )

    # Faster, but less precise
    torch.set_float32_matmul_precision("high")

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
        max_epochs=tune_settings["max_epochs"],
        devices="auto",
        accelerator="auto",
        # strategy=RayFSDPStrategy(),
        strategy=RayDDPStrategy(find_unused_parameters=True),
        # strategy=RayDeepSpeedStrategy(),
        callbacks=[RayTrainReportCallback(), lr_monitor],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
        # other trainer args
        log_every_n_steps=32,
        val_check_interval=0.25,
        precision="bf16-mixed",
        default_root_dir="./.lightning/nef-gpt",
        deterministic=True,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)



def main():
    search_space = {
        # "batch_size": tune.choice([2]),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        # maybe also some architecture hyperparameters
    }

    scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"CPU": 2, "GPU": 1})
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

    scheduler = ASHAScheduler(max_t=tune_settings["max_epochs"], grace_period=1, reduction_factor=2)

    ray.init(num_cpus=8, num_gpus=1)

    algo = BayesOptSearch(random_search_steps=tune_settings["init_tune_steps"])

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="train/loss",
            mode="min",
            num_samples=tune_settings["total_tune_steps"],
            scheduler=scheduler,
            search_alg=algo
        ),
    )
    return tuner.fit()


if __name__ == "__main__":
    main()
