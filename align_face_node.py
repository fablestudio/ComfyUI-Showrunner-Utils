import numpy as np
import torch
import cv2
import math


class AlignFace:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pose_keypoints": ("POSE_KEYPOINT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "align_face"
    CATEGORY = "Showrunner Nodes"

    def align_face(self, image, pose_keypoints):
        # pose_keypoints is now a list of dictionaries, we'll use the first one
        keypoints = pose_keypoints[0]["people"][0]["pose_keypoints_2d"]

        # Extract eye positions
        left_eye_x, left_eye_y = keypoints[14 * 3 : 14 * 3 + 2]
        right_eye_x, right_eye_y = keypoints[15 * 3 : 15 * 3 + 2]

        # Calculate the midpoint of the eyes
        mid_eye_x = (left_eye_x + right_eye_x) / 2
        mid_eye_y = (left_eye_y + right_eye_y) / 2

        # Get image dimensions
        height, width = image.shape[1:3]

        # Calculate translation to center the eyes
        dx = width // 2 - mid_eye_x
        dy = height // 2 - mid_eye_y

        # Calculate rotation angle
        angle = math.degrees(
            math.atan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x)
        )

        # Create rotation matrix around the midpoint of the eyes
        rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)

        # Apply transformations to each channel
        transformed_image = torch.zeros_like(image)
        for c in range(image.shape[0]):
            channel = image[c].numpy()

            # First, translate the image
            translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            translated_channel = cv2.warpAffine(
                channel, translation_matrix, (width, height)
            )

            # Then, rotate the image
            rotated_channel = cv2.warpAffine(
                translated_channel,
                rotation_matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
            )

            transformed_image[c] = torch.from_numpy(rotated_channel)

        return (transformed_image,)
