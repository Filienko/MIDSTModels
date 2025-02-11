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
                 normalize: Callable = None, denormalize: Callable = None,
                 feature_weights: torch.Tensor = None):
        self.eps_getter = eps_getter
        self.betas = betas.float()
        self.noise_level = torch.cumprod(1 - betas, dim=0).float()
        self.interval = interval
        self.attack_num = attack_num
        self.normalize = normalize
        self.denormalize = denormalize
        self.T = len(self.noise_level)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Feature-specific noise scaling
        self.feature_weights = feature_weights.to(self.device) if feature_weights is not None else None

    def _ensure_shape(self, x):
        x = x.to(self.device).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def get_xt_coefficient(self, step, feature_idx=None):
        base_noise_level = self.noise_level[step]
        if self.feature_weights is not None and feature_idx is not None:
            # Adjust noise level per feature
            noise_level = base_noise_level * self.feature_weights[feature_idx]
        else:
            noise_level = base_noise_level
        return noise_level ** 0.5, (1 - noise_level) ** 0.5

    def get_xt(self, x0, step, eps):
        x0 = self._ensure_shape(x0)
        eps = self._ensure_shape(eps)
        
        # Handle each feature separately
        result = torch.zeros_like(x0)
        for i in range(x0.shape[1]):
            a_T, b_T = self.get_xt_coefficient(step, i)
            result[:, i] = a_T * x0[:, i] + b_T * eps[:, i]
        return result

    def _normalize(self, x):
        return self.normalize(x) if self.normalize is not None else x

    def _denormalize(self, x):
        return self.denormalize(x) if self.denormalize is not None else x

class DDIMAttacker(Attacker):
    def get_y(self, x, step):
        result = torch.zeros_like(x)
        for i in range(x.shape[1]):
            noise_level = self.noise_level[step]
            if self.feature_weights is not None:
                noise_level = noise_level * self.feature_weights[i]
            result[:, i] = (1 / noise_level ** 0.5) * x[:, i].to(self.device)
        return result

    def get_x(self, y, step):
        result = torch.zeros_like(y)
        for i in range(y.shape[1]):
            noise_level = self.noise_level[step]
            if self.feature_weights is not None:
                noise_level = noise_level * self.feature_weights[i]
            result[:, i] = y[:, i].to(self.device) * noise_level ** 0.5
        return result

    def get_p(self, step, feature_idx=None):
        noise_level = self.noise_level[step]
        if self.feature_weights is not None and feature_idx is not None:
            noise_level = noise_level * self.feature_weights[feature_idx]
        return (1 / noise_level - 1) ** 0.5

    def get_reverse_and_denoise(self, x0, condition, step=None):
        x0 = self._normalize(x0)
        intermediates = self.ddim_reverse(x0, condition)
        intermediates_denoise = self.ddim_denoise(x0, intermediates, condition)
        return torch.stack(intermediates), torch.stack(intermediates_denoise)

    def __call__(self, x0, condition=None):
        intermediates, intermediates_denoise = self.get_reverse_and_denoise(x0, condition)
        return self.distance(intermediates, intermediates_denoise)

    def distance(self, x0, x1):
        x0 = self._ensure_shape(x0)
        x1 = self._ensure_shape(x1)
        return ((x0 - x1).abs()**2).sum(dim=-1)


class PIA(DDIMAttacker):
    def __init__(self, betas, interval, attack_num, eps_getter: EpsGetter, 
                 normalize: Callable = None, denormalize: Callable = None, 
                 lp=4, feature_weights: torch.Tensor = None,
                 feature_importance: torch.Tensor = None):
        super().__init__(betas, interval, attack_num, eps_getter, normalize, denormalize, feature_weights)
        self.lp = lp
        self.feature_importance = feature_importance.to(self.device) if feature_importance is not None else None

    def ddim_reverse(self, x0, condition):
        """
        Generate intermediates maintaining proper dimensions
        x0 shape: (batch_size, num_features)
        Returns: list of tensors each with shape (batch_size, num_features)
        """
        intermediates = []
        terminal_step = self.interval * self.attack_num
        
        # Get noise for entire input at once
        eps = self.eps_getter(x0, condition, self.noise_level, 0)
            
        for _ in reversed(range(0, terminal_step, self.interval)):
            intermediates.append(eps.clone())
        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        """
        Denoise intermediates maintaining proper dimensions
        x0 shape: (batch_size, num_features)
        intermediates: list of tensors each with shape (batch_size, num_features)
        Returns: list of tensors each with shape (batch_size, num_features)
        """
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num
        
        for idx, step in enumerate(range(self.interval, terminal_step + self.interval, self.interval)):
            eps = intermediates[idx]
            xt = self.get_xt(x0, step, eps)
            
            # Get noise for entire input at once
            eps_back = self.eps_getter(xt, condition, self.noise_level, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise

    def distance(self, x0, x1):
        """
        Calculate distance maintaining batch dimension
        x0, x1 shape: (num_steps, batch_size, num_features) or (batch_size, num_features)
        Returns: (batch_size,) tensor of distances
        """
        x0 = self._ensure_shape(x0)
        x1 = self._ensure_shape(x1)
        
        if x0.dim() == 3:  # (num_steps, batch_size, num_features)
            diff = (x0 - x1).abs()**self.lp
            if self.feature_importance is not None:
                diff = diff * self.feature_importance.view(1, 1, -1)
            return diff.sum(dim=2).sum(dim=0)
        else:  # (batch_size, num_features)
            diff = (x0 - x1).abs()**self.lp
            if self.feature_importance is not None:
                diff = diff * self.feature_importance.view(1, -1)
            return diff.sum(dim=1)

    def __call__(self, x0, condition=None):
        """
        Main call maintaining batch dimension throughout
        x0 shape: (batch_size, num_features)
        Returns: (batch_size,) tensor of distances
        """
        intermediates, intermediates_denoise = self.get_reverse_and_denoise(x0, condition)
        return self.distance(intermediates, intermediates_denoise)
    
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
