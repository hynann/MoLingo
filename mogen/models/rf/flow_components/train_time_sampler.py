import torch


class TrainTimeSampler:
    def __init__(
        self,
        distribution: str = "uniform",
    ):
        self.distribution = distribution

    @staticmethod
    def u_shaped_t(num_samples, alpha=4.0):
        alpha = torch.tensor(alpha, dtype=torch.float32)
        u = torch.rand(num_samples)
        t = (
            -torch.log(1 - u * (1 - torch.exp(-alpha))) / alpha
        )  # inverse cdf = torch.log(u * (torch.exp(torch.tensor(a)) - 1) / a) / a
        t = torch.cat([t, 1 - t], dim=0)
        t = t[torch.randperm(t.shape[0])]
        t = t[:num_samples]
        return t

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Sample time tensor for training

        Returns:
            torch.Tensor: Time tensor, shape (batch_size,)
        """
        if self.distribution == "uniform":
            t = torch.rand((batch_size,)).to(device=device, dtype=dtype)
        elif self.distribution == "lognormal":
            t = torch.sigmoid(torch.randn((batch_size,))).to(device=device, dtype=dtype)
        elif self.distribution == "u_shaped":
            t = self.u_shaped_t(batch_size).to(device=device, dtype=dtype)
        else:
            raise NotImplementedError(
                f"Time distribution '{self.dist}' is not implemented."
            )

        return t
