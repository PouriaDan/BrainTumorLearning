import torch
from torchvision.transforms import v2 as transforms

class SynchronizedTransform:
    def __init__(self, size=(256, 256), crop_size=(224, 224), randomize=True):
        self.randomize = randomize
        self.resize = transforms.Resize(size)
        self.random_crop = transforms.RandomCrop(crop_size) if randomize else None
        self.horizontal_flip = transforms.RandomHorizontalFlip() if randomize else None
        self.vertical_flip = transforms.RandomVerticalFlip() if randomize else None
        self.to_image = transforms.ToImage()
        self.to_dtype = transforms.ToDtype(torch.float32, scale=True)

    def __call__(self, img, mask):
        img = self.resize(img)
        mask = self.resize(mask)

        if self.randomize:
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.random_crop.size)
            img = transforms.functional.crop(img, i, j, h, w)
            mask = transforms.functional.crop(mask, i, j, h, w)

            img = self.horizontal_flip(img)
            mask = self.horizontal_flip(mask)

            img = self.vertical_flip(img)
            mask = self.vertical_flip(mask)

        img = self.to_image(img)
        img = self.to_dtype(img)

        mask = self.to_image(mask)
        mask = (mask > 0.5).float()

        return img, mask

