import torch


def decorator_timer(some_function):
    from time import time

    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        end = time() - t1
        return result, end

    return wrapper


def get_default_device(cpu_only=False):
    """
    Return either mps or cuda depending on the availability of the GPU
    Fall back to cpu if no GPU is available
    """
    if torch.cuda.is_available() and not cpu_only:
        return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #    return torch.device("mps")
    else:
        return torch.device("cpu")
    

from typing import List, Tuple, Union
import os
import torch
from collections import OrderedDict
import numpy as np

def ensure_folder_exists(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def make_coordinates(
    shape: Union[Tuple[int], List[int]],
    bs: int,
    coord_range: Union[Tuple[int], List[int]] = (0, 1),
) -> torch.Tensor:
    x_coordinates = np.linspace(coord_range[0], coord_range[1], shape[0])
    y_coordinates = np.linspace(coord_range[0], coord_range[1], shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    coordinates = np.stack([x_coordinates, y_coordinates]).T
    coordinates = np.repeat(coordinates[np.newaxis, ...], bs, axis=0)
    return torch.from_numpy(coordinates).type(torch.float)

def reconstruct_image_correct_and_not_luis_bs(model: torch.nn.Module, image_size: tuple = (28, 28)):

    input_coords = make_coordinates(image_size, 1)
     # Generate image using the INR model
    with torch.no_grad():
        reconstructed_image = model(input_coords)
        reconstructed_image = torch.sigmoid(reconstructed_image)
        reconstructed_image = reconstructed_image.view(*image_size, -1)
        reconstructed_image = reconstructed_image.permute(2, 0, 1)

    return reconstructed_image

def reconstruct_image(model: torch.nn.Module, image_size: tuple = (28, 28)):

    input_coords = make_coordinates(image_size, 1)
     # Generate image using the INR model
    with torch.no_grad():
        reconstructed_image = model(input_coords)
        reconstructed_image = torch.sigmoid(reconstructed_image)
        reconstructed_image = reconstructed_image.view(*image_size, -1)
        reconstructed_image = reconstructed_image.permute(2, 0, 1)

    return reconstructed_image.squeeze(0).numpy()

def state_dict_to_min_max(state_dict: OrderedDict):
    weights = []

    for key, value in state_dict.items():
        weights.append(value)

    vmax = torch.Tensor([w.max() for w in weights]).max()
    vmin = torch.Tensor([w.min() for w in weights]).min()

    return vmin, vmax


def get_vmin_vmax(image_idx: int, num_epochs: int, foldername: str, comparison_model: OrderedDict = None):
    
    vmins = []
    vmaxs = []
    for epoch in range(num_epochs):        
        model_path = "{}/image-{}".format(foldername, image_idx) + f"_model_epoch_{epoch}.pth"
        assert os.path.exists(model_path), f"File {model_path} does not exist"

        if comparison_model:
            new_state_dict, _, _ = get_model_difference_dict(torch.load(model_path), comparison_model)
            vmin, vmax = state_dict_to_min_max(new_state_dict)
        else:
            vmin, vmax = state_dict_to_min_max(torch.load(model_path))
        vmins.append(vmin)
        vmaxs.append(vmax)

    vmax = torch.Tensor([v.max() for v in vmaxs]).max()
    vmin = torch.Tensor([v.min() for v in vmins]).min()

    return vmin, vmax

def get_model_difference(model_1: torch.nn.Module, model_2: torch.nn.Module, untrained_model: torch.nn.Module):

    state_dict, _, _ = get_model_difference_dict(model_1.state_dict(), model_2.state_dict())

    untrained_model.load_state_dict(state_dict)

    return untrained_model

def get_model_difference_dict(model_1_dict: OrderedDict, model_2_dict: OrderedDict):
    new_state_dict = OrderedDict()
    assert model_1_dict.keys() == model_2_dict.keys(), "Model keys do not match"
    for key in model_1_dict.keys():
        new_state_dict[key] = model_1_dict[key] - model_2_dict[key]

    return new_state_dict, model_1_dict, model_2_dict

def backtransform_weights(flattened_weights, original_weights_dict):
    reconstructed_dict = OrderedDict()
    start = 0
    for key, tensor in original_weights_dict.items():
        num_elements = tensor.numel()
        flattened_slice = flattened_weights[0, start:start + num_elements]
        reconstructed_tensor = flattened_slice.view(tensor.shape)
        reconstructed_dict[key] = reconstructed_tensor
        start += num_elements
    return reconstructed_dict


    
