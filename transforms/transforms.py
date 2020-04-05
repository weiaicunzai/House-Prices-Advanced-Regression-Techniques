import cv2

class Resize:

    def __init__(self, height=None, width=None):
        """resize image(an numpy array) to given height and
        width, if only height or width were given, resize
        height or width to given length while keeping aspect
        ratio.
        Args:
            height: None or int, image height size
            width: None or int, image width size
        """
        self.height = height
        self.width = width
        if self.height is None and self.width is None:
            raise ValueError(('height and width can not be'
                'None at the same time'))


    def __call__(self, image):

        height, width = image.shape[:2]
        if self.width is None:
            new_width = int(width * self.height / height)
            new_height = self.height

        elif self.height is None:
            new_height = int(height * self.width / width)
            new_width = self.width

        else:
            new_height = self.height
            new_width = self.width

        image = cv2.resize(image, (new_width, new_height))

        #print(image.shape)

        return image
