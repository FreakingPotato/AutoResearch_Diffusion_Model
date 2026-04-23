"""Noise schedule for diffusion training."""
import torch


class UniformSchedule:
    """Uniform masking schedule."""
    def __init__(self, noise_steps=128, mask_id=1):
        self.noise_steps = noise_steps
        self.mask_id = mask_id

    def sample_t(self, batch_size, device):
        return torch.randint(1, self.noise_steps+1, (batch_size,), device=device)

    def forward_process(self, x, t):
        """Add noise (mask tokens) to x based on timestep t."""
        prob = t.float() / self.noise_steps
        mask = torch.rand_like(x, dtype=torch.float) < prob.unsqueeze(1)
        xn = x.clone()
        xn[mask] = self.mask_id
        return xn, mask
