from lightning.pytorch.cli import LightningCLI

from data.loader import ShapeNetData
from nef_gpt import NefGPT


def cli_main():
    cli = LightningCLI(NefGPT, ShapeNetData)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
