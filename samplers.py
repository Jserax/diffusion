from typing import Optional, Union

import torch


class EulerSampler:
    def __init__(
        self,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
        device: str = "cpu",
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_train_timesteps, device=device
            )
        elif beta_schedule == "scaled_linear":
            self.betas = (
                torch.linspace(
                    beta_start**0.5, beta_end**0.5, num_train_timesteps, device=device
                )
            ) ** 2
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.train_sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.train_timesteps = torch.linspace(
            0,
            self.num_train_timesteps - 1,
            self.num_train_timesteps,
            dtype=torch.int32,
            device=self.device,
        ).flip(0)
        self.initial_scale = self.train_sigmas.max()

    def scale_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigmas = self.train_sigmas[timestep]
        while len(sigmas.shape) < len(sample.shape):
            sigmas = sigmas.unsqueeze(-1)
        sample = sample / ((sigmas**2 + 1) ** 0.5)
        return sample

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.inference_timesteps = torch.linspace(
            0,
            self.num_train_timesteps - 1,
            num_inference_steps,
            dtype=torch.int32,
            device=self.device,
        ).flip(0)

        self.inference_sigmas = torch.cat(
            (self.train_sigmas[self.inference_timesteps], torch.tensor([0]))
        )
        self.initial_scale = self.inference_sigmas.max()

    @torch.no_grad()
    def step(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        step: Union[int, torch.Tensor],
        s_churn: Optional[float] = None,
    ) -> torch.Tensor:
        timestep = self.inference_timesteps[step]
        if s_churn is not None:
            gamma = min(s_churn / (len(self.inference_timesteps) - 1), 2**0.5 - 1)
            eps = torch.rand_like(sample)
            sigma = self.inference_sigmas[timestep]
            sample = sample + eps * ((sigma * (gamma + 1)) ** 2 - sigma**2) ** 0.5
        dt = self.inference_sigmas[timestep + 1] - self.inference_sigmas[timestep]
        sample = sample + dt * model_output
        return sample

    def add_noise(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        scale_input: bool = True,
    ) -> torch.FloatTensor:
        if not sample.device == self.device:
            sample = sample.to(self.device)
        if not noise.device == self.device:
            noise = noise.to(self.device)
        if not timestep.device == self.device:
            timestep = timestep.to(self.device)
        sigmas = self.train_sigmas[timestep]
        while len(sigmas.shape) < len(sample.shape):
            sigmas = sigmas.unsqueeze(-1)
        noisy_sample = sample + sigmas * noise
        if scale_input:
            return self.scale_input(noisy_sample, timestep)
        else:
            return noisy_sample
