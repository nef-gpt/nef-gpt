from data.loader import ShapeNetData, ShapeNetDataConfig
from nef_gpt import NefGPT, NanoGPTConfig, OptimizerConfig, LRConfig
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger


def main():

    vq_path = "./weights/rq_search_results/shapenet_retrained_learnable_rq_model_dim_1_vocab_127_batch_size_16_threshold_ema_dead_code_0_kmean_iters_1_num_quantizers_1_use_init_False.pth"
    data = ShapeNetData(
        data_config=ShapeNetDataConfig("./datasets/shapenet_nef_2/pretrained", vq_path)
    )

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

    trainer = Trainer(
        max_epochs=5,
        # Used to limit the number of batches for testing and initial overfitting
        limit_train_batches=8,
        limit_val_batches=2,
        # Logging stuff
        log_every_n_steps=2,
        logger=logger,
        profiler="simple",
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
