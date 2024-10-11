import math
import random
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from src.utils.lp_render.camera import get_rotation_matrix


def minmax_normalization(x):
    return (x - x.min()) / (x.max() - x.min())

def ease_in_out_sine(t):
    return -(torch.cos(math.pi * t) - 1) / 2

class SyntheticHeadPoseGeneration:
    def __init__(self, device, fps, window_size, threshold):
        self.device = device
        self.fps = fps
        self.window_size = window_size
        self.threshold = threshold

    def __call__(self, lower_upper_lip_expressions):
        weight_list = self.get_lip_based_weight_list(lower_upper_lip_expressions=lower_upper_lip_expressions)

        pitch_values = self.generate_head_moves(lower_upper_lip_expressions.shape[0],
                                                degree=random.uniform(0., 3.), #degree_hash["pitch_degree"],
                                                frequency=random.uniform(.5, 1.5)) * weight_list
        yaw_values = self.generate_head_moves(lower_upper_lip_expressions.shape[0],
                                                degree=random.uniform(0., 3.), #degree_hash["yaw_degree"],
                                                frequency=random.uniform(.5, 1.5)) * weight_list
        roll_values = self.generate_head_moves(lower_upper_lip_expressions.shape[0],
                                                degree=random.uniform(0., 3.), #degree_hash["roll_degree"],
                                                frequency=random.uniform(.5, 1.5)) * weight_list

        R = get_rotation_matrix(pitch_values, yaw_values, roll_values)
        return R.to(self.device)

    def smooth_weight_list(self, weight_list):
        pad_length = self.window_size // 2
        kernel = torch.ones(self.window_size) / self.window_size
        padded_input = F.pad(weight_list.unsqueeze(0), (pad_length, pad_length), value=1)
        smoothed_weight_list = F.conv1d(padded_input.unsqueeze(0),
                                        kernel.view(1, 1, -1),
                                        groups=1).squeeze(0).squeeze(0)
        return smoothed_weight_list

    def get_lip_based_weight_list(self, lower_upper_lip_expressions):
        normalized_lip_based_exp = minmax_normalization(lower_upper_lip_expressions)

        derivative_lip_based_exp = torch.diff(normalized_lip_based_exp, dim=0)
        derivative_lower_lip_values = derivative_lip_based_exp[:, 0].abs().mean(dim=1)
        derivative_upper_lip_values = derivative_lip_based_exp[:, 1].abs().mean(dim=1)

        weight_list = torch.ones(derivative_lip_based_exp.shape[0])
        weight_list[(derivative_lower_lip_values<self.threshold)&(derivative_upper_lip_values<self.threshold)] = 0
        weight_list = torch.cat([weight_list, torch.zeros(1)], dim=0)

        smoothed_weight_list = self.smooth_weight_list(weight_list)
        return smoothed_weight_list

    def generate_head_moves(self, num_frames, degree, frequency):
        t = torch.linspace(0, int(num_frames/self.fps), num_frames)
        if t.shape[0] < num_frames:
            t = torch.cat([t, t[-1].unsqueeze(0).repeat(1, num_frames-t.shape[0])], dim=-1)
        else:
            t = t[:num_frames]

        eased_t = ease_in_out_sine(t)

        base_movement = degree * torch.sin(2 * math.pi * frequency * t)
        micro_movement = degree * 0.05 * torch.randn_like(t)
        values = base_movement * eased_t + micro_movement
        values = torch.clamp(values, -degree, degree)

        return values