import torch
from typing import Callable, Optional

class EpsGetter:
    def __init__(self, model):
        self.model = model

    def __call__(self, xt: torch.Tensor, condition: torch.Tensor = None, 
                 noise_level=None, t: Optional[int] = None) -> torch.Tensor:
        if len(xt.shape) == 1:
            xt = xt.unsqueeze(0)
        
        batch_size = xt.shape[0]
        t_tensor = torch.ones((batch_size,), device=xt.device, dtype=torch.long) * t

        diffusion = self.model[(None, 'trans')]['diffusion']
        eps_pred = diffusion._denoise_fn(xt, t_tensor)

        return eps_pred
    
    def get_eps_and_var(self, xt: torch.Tensor, t: int) -> tuple:
        eps = self.__call__(xt, t=t)
        return eps, None

class Attacker:
    def __init__(self, betas, interval, attack_num, eps_getter: EpsGetter, 
                 normalize: Callable = None, denormalize: Callable = None):
        self.eps_getter = eps_getter
        self.betas = betas.float()
        self.noise_level = torch.cumprod(1 - betas, dim=0).float()
        self.interval = interval
        self.attack_num = attack_num
        self.normalize = normalize
        self.denormalize = denormalize
        self.T = len(self.noise_level)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_shape(self, x):
        x = x.to(self.device).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def get_xt_coefficient(self, step):
        return self.noise_level[step] ** 0.5, (1 - self.noise_level[step]) ** 0.5

    def get_xt(self, x0, step, eps):
        if len(x0.shape) == 1:
            x0 = x0.unsqueeze(0)
        if len(eps.shape) == 1:
            eps = eps.unsqueeze(0)
            
        a_T, b_T = self.get_xt_coefficient(step)
        return a_T * x0 + b_T * eps

    def __call__(self, x0, xt, condition):
        raise NotImplementedError

    def _normalize(self, x):
        return self.normalize(x) if self.normalize is not None else x

    def _denormalize(self, x):
        return self.denormalize(x) if self.denormalize is not None else x

class DDIMAttacker(Attacker):
    def get_y(self, x, step):
        return (1 / self.noise_level[step] ** 0.5) * x.to(self.device)

    def get_x(self, y, step):
        return y.to(self.device) * self.noise_level[step] ** 0.5

    def get_p(self, step):
        return (1 / self.noise_level[step] - 1) ** 0.5

    def get_reverse_and_denoise(self, x0, condition, step=None):
        x0 = self._normalize(x0)
        intermediates = self.ddim_reverse(x0, condition)
        intermediates_denoise = self.ddim_denoise(x0, intermediates, condition)
        return torch.stack(intermediates), torch.stack(intermediates_denoise)

    def __call__(self, x0, condition=None):
        intermediates, intermediates_denoise = self.get_reverse_and_denoise(x0, condition)
        return self.distance(intermediates, intermediates_denoise)

    def distance(self, x0, x1):
        x0 = x0.unsqueeze(0) if len(x0.shape) == 1 else x0
        x1 = x1.unsqueeze(0) if len(x1.shape) == 1 else x1
        return ((x0 - x1).abs()**2).sum(dim=-1)

    def ddim_reverse(self, x0, condition):
        raise NotImplementedError

    def ddim_denoise(self, x0, intermediates, condition):
        raise NotImplementedError

class PIA(DDIMAttacker):
    def __init__(self, betas, interval, attack_num, eps_getter: EpsGetter, 
                 normalize: Callable = None, denormalize: Callable = None, lp=4):
        super().__init__(betas, interval, attack_num, eps_getter, normalize, denormalize)
        self.lp = lp

    def distance(self, x0, x1):
        # print("x0", x0)
        # print("x1", x1)
        distance = ((x0 - x1).abs()**self.lp).flatten(1).sum(dim=-1)
        # print("distance", distance)
        return distance

    def ddim_reverse(self, x0, condition):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        eps = self.eps_getter(x0, condition, self.noise_level, 0)
        for _ in reversed(range(0, terminal_step, self.interval)):
            intermediates.append(eps)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(self.interval, terminal_step + self.interval, self.interval)):
            eps = intermediates[idx]
            eps_back = self.eps_getter(self.get_xt(x0, step, eps), condition, self.noise_level, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise
    

class PIAN(DDIMAttacker):
    def __init__(self, betas, interval, attack_num, eps_getter: EpsGetter, normalize: Callable = None, denormalize: Callable = None, lp=4):
        super().__init__(betas, interval, attack_num, eps_getter, normalize, denormalize)
        self.lp = lp

    def distance(self, x0, x1):
        distance = ((x0 - x1).abs()**self.lp).flatten(1).sum(dim=-1)
        # print("distance", distance)
        return distance

    def ddim_reverse(self, x0, condition):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        eps = self.eps_getter(x0, condition, self.noise_level, 0)
        eps = eps / eps.abs().mean(list(range(1, eps.ndim)), keepdim=True) * (2 / torch.pi) ** 0.5
        for _ in reversed(range(0, terminal_step, self.interval)):
            intermediates.append(eps)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(self.interval, terminal_step + self.interval, self.interval)):
            eps = intermediates[idx]

            eps_back = self.eps_getter(self.get_xt(x0, step, eps), condition, self.noise_level, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise
