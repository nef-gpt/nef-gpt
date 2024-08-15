from data.loader import ShapeNetData, ShapeNetDataConfig
from nef_gpt import NefGPT, NanoGPTConfig, OptimizerConfig, LRConfig
from lightning import Trainer


def main():

    vq_path = "./weights/rq_search_results/shapenet_retrained_learnable_rq_model_dim_1_vocab_127_batch_size_16_threshold_ema_dead_code_0_kmean_iters_1_num_quantizers_1_use_init_False.pth"
    data = ShapeNetData(
        data_config=ShapeNetDataConfig("./datasets/shapenet_nef_2/pretrained", vq_path)
    )

    model = NefGPT(
        config=NanoGPTConfig.from_preset("small"),
        optimizer_config=OptimizerConfig(),
        lr_config=LRConfig(),
    )

    trainer = Trainer(max_epochs=100)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
