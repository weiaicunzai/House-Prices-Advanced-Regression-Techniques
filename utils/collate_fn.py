import torch
import torch.nn.functional as F


class PadBatch:

    def __init__(self, image_width=100, image_height=32):
        """pad a batch of images to given size

        Args:
            image_width: minimum output width,
            final output width is max(max width of a batch
            of images, image_width)
            image_height: minimum output height, final output
            height is max(max height of a batch of images, image_height)
        """
        self.image_width = image_width
        self.image_height = image_height

    def _pad(self, image, max_width, max_height):

        height, width = image.shape[1:]
        left_pad = (max_width - width) // 2
        right_pad = max_width - width - left_pad
        top_pad = (max_height - height) // 2
        bot_pad = max_height - height - top_pad
        image = F.pad(image, (left_pad, right_pad, top_pad, bot_pad))

        return image

    def __call__(self, batch):
        """input images should be a list of tensors with CHW format"""
        images, labels, lengths = zip(*batch)

        max_width = max(
            max(image.shape[2] for image in images),
            self.image_width
        )

        max_height = max(
            max(image.shape[1] for image in images),
            self.image_height
        )

        images = [self._pad(image, max_width, max_height) for image in images]
        images = torch.stack(images)

        labels = [character for label in labels for character in label]
        labels = torch.tensor(labels)

        lengths = torch.tensor(lengths)

        return images, labels, lengths