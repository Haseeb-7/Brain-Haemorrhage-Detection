import torch
import cv2
import numpy as np

from modules.model_handler import model, DEVICE


class GradCAM:

    def __init__(self):

        self.activations = None
        self.gradients = None

        target_layer = model.layers[-1].blocks[-1].norm2

        target_layer.register_forward_hook(
            self.save_activation
        )

        target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, inp, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):

        input_tensor = input_tensor.to(DEVICE)

        model.zero_grad()

        logits = model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(logits)

        score = logits[:, class_idx]

        score.backward()

        activations = self.activations
        gradients = self.gradients

        weights = gradients.mean(dim=1, keepdim=True)

        cam = (weights * activations).sum(dim=-1)

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()[0]

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        cam = cv2.resize(
            cam,
            (256, 256)
        )

        return cam


gradcam = GradCAM()