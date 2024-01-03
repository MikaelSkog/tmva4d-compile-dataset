from functools import reduce

import cv2
import numpy as np
from ultralytics import YOLO

class MaskPredictor:
    def __init__(self, model_path):
        # Load the model.
        self.model = YOLO(model_path) 

        # Set (some of) the model parameters.
        self.model.conf = 0.1  # NMS confidence threshold.
        self.model.iou = 0.5  # NMS IoU threshold.
        self.model.agnostic = False  # NMS class-agnostic.
        self.model.multi_label = False  # NMS multiple labels per box.
        self.model.max_det = 100  # Maximum number of detections per image.
        self.model.classes = [0]  # Only make detections for class 'person'.

    def predict(self, image):
        # Predict the mask of the image for the 'person' class.
        results = self.model.predict(source=image, save=False, classes=0, verbose=False)

        # Get the masks for the 'person' class.
        if results[0].masks is None:
            person_masks = np.zeros((128, 128))
        else:
            person_masks = results[0].masks.data.cpu().numpy()

        # Combine the masks of the 'person' class.
        combined_person_mask = reduce(np.logical_or, person_masks).astype(np.uint8)

        # Resize the combined mask for the 'person' class.
        combined_person_mask_resized = cv2.resize(combined_person_mask, dsize=(128, 128), interpolation=cv2.INTER_NEAREST_EXACT)

        # Get the background mask, which is the combined resized 'person' mask inverted).
        background_mask = 1 - combined_person_mask_resized

        # Combine the masks for the 'background' and 'person' classes into a single NumPy array.
        mask_both_classes = np.array([background_mask, combined_person_mask_resized])

        # Convert the masks to float64.
        mask_both_classes_float64 = mask_both_classes.astype(np.float64)

        # Get a visualization segmentation masks, overlayed on top of the image.

        image_with_masks = results[0].plot()

        return image_with_masks, mask_both_classes_float64