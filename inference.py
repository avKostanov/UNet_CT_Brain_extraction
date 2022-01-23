import numpy as np
import torch
from skimage.measure import label
from dpipe.im.slices import iterate_axis
from .model import BrainSegModel


class BrainSegmenter:

    def __init__(
        self,
        weightsPath,
        device=torch.device('cpu'),
        threshold=0.8,
    ):
        self.weightsPath = weightsPath
        self.device = device
        self.threshold = threshold
        self.model = BrainSegModel()

        self.model.load_state_dict(torch.load(self.weightsPath))
        self.model.eval().to(self.device)

    def __call__(self, image):
        if not np.allclose(image.min(), -1024, atol=100):
            image = image - 1024
        image = np.clip(image, 0, 80)
        result = np.zeros_like(image)

        for i, slc in enumerate(iterate_axis(image, 2)):
            slc = torch.from_numpy(slc).unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = self.model(slc.float().unsqueeze(0)) > self.threshold
                preds = preds.squeeze(0).squeeze(0).cpu().detach().numpy()
            result[:, :, i] = preds

        labels = label(result)
        assert labels.max() != 0  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC
