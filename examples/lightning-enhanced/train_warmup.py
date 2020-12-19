import math
from typing import Optional
from itertools import chain

import typer
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_helper_bot.optimizers import RAdam
from byol_pytorch.byol_pytorch import set_trainable

from bit_models import KNOWN_MODELS as BIT_MODELS
from dataset import get_datasets, IMAGE_SIZE
from train import SelfSupervisedLearner


class FirstStageLearner(SelfSupervisedLearner):
    def configure_optimizers(self):
        optimizer = RAdam([
            {
                "params": chain.from_iterable([
                    x.parameters() for x in [
                        self.learner.online_encoder.projector,
                        self.learner.online_predictor
                    ]]),
                "lr": self.learning_rate
            }
        ])
        return {
            'optimizer': optimizer,
        }


def main(
        arch: str, image_folder: str,
        batch_size: Optional[int] = None,
        from_model: Optional[str] = None,
        grad_accu: int = 1, steps: Optional[int] = None,
        num_gpus: int = 1, epochs: int = 1, lr: float = 4e-4):
    if arch.startswith("BiT"):
        base_model = BIT_MODELS[arch](head_size=-1)
        print("Loading pretrained model...")
        base_model.load_from(np.load(f"cache/pretrained/{arch}.npz"))
        net_final_size = base_model.width_factor * 2048
    else:
        raise ValueError(f"arch '{arch}'' not supported")
    train_ds, valid_ds = get_datasets(image_folder, val_ratio=0.05)

    set_trainable(base_model, False)
    model = FirstStageLearner(
        base_model,
        train_ds,
        valid_ds,
        epochs,
        lr,
        num_gpus=num_gpus,
        batch_size=batch_size if batch_size else 4,
        image_size=IMAGE_SIZE,
        projection_size=256,
        projection_hidden_size=4096,
        net_final_size=net_final_size,
        use_momentum=False
    )

    if steps:
        trainer = pl.Trainer(
            accelerator='ddp' if num_gpus > 1 else None,
            # amp_backend="apex", amp_level='O2',
            # precision=16,
            gpus=num_gpus,
            val_check_interval=0.5,
            gradient_clip_val=10,
            max_steps=steps,
            accumulate_grad_batches=grad_accu,
            auto_scale_batch_size='power' if batch_size is None else None
        )
    else:
        trainer = pl.Trainer(
            accelerator='ddp' if num_gpus > 1 else None,
            # amp_backend="apex", amp_level='O2',
            # precision=16,
            gpus=num_gpus,
            val_check_interval=0.5,
            gradient_clip_val=10,
            max_epochs=epochs,
            accumulate_grad_batches=grad_accu,
            auto_scale_batch_size='power' if batch_size is None else None
        )

    if batch_size is None:
        trainer.tune(model)

    trainer.fit(model)

    if num_gpus == 1 or torch.distributed.get_rank() == 0:
        torch.save({
            "online_encoder_proj": model.learner.online_encoder.projector.state_dict(),
            "online_encoder_net": model.learner.online_encoder.net.state_dict(),
            "online_predictor": model.learner.online_predictor.state_dict(),
            "target_encoder_net": model.learner.target_encoder.net.state_dict(),
            "target_encoder_proj": model.learner.target_encoder.projector.state_dict(),
            "config": {
                "arch": arch
            }
        }, f"cache/byol_{arch}_warmed_up.pth")
        print("Model saved")


if __name__ == '__main__':
    typer.run(main)
