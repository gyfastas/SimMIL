import torchvision.transforms as transforms
from .cls_runner import ClsRunner

class LinClsRunner(ClsRunner):
    def build_train_augmentation(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        train_augmentation = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        return train_augmentation
