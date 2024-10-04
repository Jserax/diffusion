from typing import Any, Callable, Dict, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from samplers import EulerSampler
from util import denormalize


class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        diffusion_model: torch.nn.Module,
        text_model: BertModel,
        tokenizer: BertTokenizerFast,
        sampler: EulerSampler,
        criterion: Union[torch.nn.Module, Callable],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        train_metrics: torch.nn.ModuleDict,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.diffusion_model = diffusion_model
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.sampler = sampler
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_metrics = train_metrics
        self.save_hyperparameters(
            ignore=["diffusion_model", "text_model", "criterion", "train_metrics"]
        )

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        images, tokens, attn_mask = batch
        noise = torch.randn_like(images, device=images.device)
        timestep = torch.randint(0, 1000, size=images.size(), device=images.device)
        input = self.sampler.add_noise(images, noise, timestep)
        with torch.no_grad():
            context = self.text_model(tokens, attn_mask).last_hidden_state
        pred_noise = self.diffusion_model(input, context, timestep)
        train_loss = self.criterion(pred_noise, noise)
        for metric_name, metric in self.train_metrics.items():
            metric.update(pred_noise, noise)
            self.log(
                f"train_{metric_name}",
                metric,
                on_epoch=True,
                on_step=False,
                logger=True,
            )
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )
        return train_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.optimizer(
            [
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.diffusion_model.parameters()
                    ),
                    "name": "diffusion_model",
                }
            ]
        )
        scheduler = self.scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def diffusion_loop(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        img_shape: tuple[int, int] = (64, 64),
        s_churn: Optional[float] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed=seed)
        self.sampler.set_timesteps(num_inference_steps)
        sample = (
            torch.randn(1, 3, img_shape[0], img_shape[1]) * self.sampler.initial_scale
        )

        with torch.no_grad():
            tokenizer_output = self.tokenizer.batch_encode_plus(
                [prompt, ""] if cfg_scale is not None else [prompt],
                add_special_tokens=True,
                # max_length=64,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                return_token_type_ids=False,
            )
            tokens = tokenizer_output["input_ids"]
            attn_mask = tokenizer_output["attention_mask"]
            context = self.text_model(tokens, attn_mask).last_hidden_state
        for step, timestep in enumerate(self.sampler.inference_timesteps):
            model_input = torch.cat([sample] * 2) if cfg_scale is not None else sample
            model_input = self.sampler.scale_input(model_input, timestep)
            pred_noise = self.diffusion_model(model_input, context, step)
            if cfg_scale is not None:
                pred_noise_uncond, pred_noise_text = pred_noise.chunk(2)
                pred_noise = pred_noise_uncond + cfg_scale * (
                    pred_noise_text - pred_noise_uncond
                )
            sample = self.sampler.step(sample, pred_noise, step, s_churn)

        return denormalize(sample)
