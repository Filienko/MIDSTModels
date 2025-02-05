import numpy as np
import pandas as pd
import torch
import os



class Attacker:
    def __init__(
        self,
        diffusion,
        train_iter,
        steps,
        device=torch.device("cuda:1"),
        attacker=None,
        save_dir='./',
    ):
        self.diffusion = diffusion
        self.train_iter = train_iter
        self.steps = steps
        self.device = device
        self.attacker = attacker
        self.save_dir = save_dir

    def run_attack(self):
        step = 0
        saved_outputs = {}
        while step < self.steps:
            x, _ = next(self.train_iter)
            distance = self.diffusion.attack(x, attacker=self.attacker)
            # model_out = self._denoise_fn(x_in, t)
            saved_outputs[step] = distance.clone().detach().cpu()
            print(f"Step {step} done")
        torch.save(saved_outputs, os.path.join(self.save_dir, "train_saved_outputs_new_distances.pth"))
