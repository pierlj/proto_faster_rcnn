import json
import os

class DatasetMeta():
    def __init__(self, name, path, categories_path, n_classes, annot_type, mean, std, min_size=800, max_size=1333, prior=None):
        self.name = name
        self.path = path
        self.categories_path = categories_path
        self.n_classes = n_classes
        self.annot_type = annot_type # Mostly 'DOTA
        self.mean = mean
        self.std = std
        self.min_size = min_size
        self.max_size = max_size
        if type(prior) == str and os.path.isfile(prior):
            self.prior = self.get_prior_from_file(prior)
        elif type(prior) == dict:
            self.prior = prior
        else:
            self.prior = [1 / n_classes for _ in range(n_classes)]
    
    def __str__(self):
        return self.name
    
    def to_dict(self):
        return self.__dict__
    
    def get_prior_from_file(self, file_name):
        with open(file_name, 'r') as f:
            instances = json.load(f)
        prior = {}
        total = sum(instances.values())
        for k, v in instances.items():
            prior[int(k)] = v / total
        return prior


mnist_meta = DatasetMeta(
    'MNIST', 
    '/home/pierre/Documents/PHD/Datasets/MNIST/prepared/train/',
    '/home/pierre/Documents/PHD/Datasets/MNIST/prepared/train/categories.json',
    10, 
    'DOTA', (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081), min_size=512, max_size=512)

mnist_meta_test = DatasetMeta(
    'MNIST', 
    '/home/pierre/Documents/PHD/Datasets/MNIST/prepared/test/',
    '/home/pierre/Documents/PHD/Datasets/MNIST/prepared/test/categories.json',
    10, 
    'DOTA', (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081), min_size=512, max_size=512)

mnist_cla_meta = DatasetMeta(
    'MNIST',
    '/home/pierre/Documents/PHD/Datasets/MNIST_CLA/train/',
    '/home/pierre/Documents/PHD/Datasets/MNIST_CLA/train/categories.json',
    10,
    'DOTA', (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081), min_size=256, max_size=256)

omni_meta = DatasetMeta(
    'MNIST',
    '/home/pierre/Documents/PHD/Datasets/Omniglot/prepared/train/',
    '/home/pierre/Documents/PHD/Datasets/Omniglot/prepared/train/categories.json',
    964,
    'DOTA', (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081), min_size=512, max_size=512)

vhr_meta = DatasetMeta(
    'VHR',
    '/home/pierre/Documents/PHD/Datasets/VHR_10/train/',
    '/home/pierre/Documents/PHD/Datasets/VHR_10/categories.json',
    10,
    # 'DOTA', (102.9801, 115.9465, 122.7717),(1.0, 1.0, 1.0), # use for FCOS
    # 'DOTA', (0.3782, 0.3816, 0.3579),(0.1700, 0.1659, 0.1613),
    'DOTA', (0.485, 0.456, 0.406),(0.229, 0.224, 0.225),
    min_size=384, max_size=512)

vhr_meta_test = DatasetMeta(
    'VHR',
    '/home/pierre/Documents/PHD/Datasets/VHR_10/test/',
    '/home/pierre/Documents/PHD/Datasets/VHR_10/categories.json',
    10,
    # 'DOTA', (102.9801, 115.9465, 122.7717),(1.0, 1.0, 1.0), # use for FCOS
    # 'DOTA', (0.3782, 0.3816, 0.3579),(0.1700, 0.1659, 0.1613),
    'DOTA', (0.485, 0.456, 0.406),(0.229, 0.224, 0.225),
    min_size=384, max_size=512)

vhr_meta_full = DatasetMeta(
    'VHR',
    '/home/pierre/Documents/PHD/Datasets/VHR_10/full/train/',
    '/home/pierre/Documents/PHD/Datasets/VHR_10/full/categories.json',
    10,
    # 'DOTA', (102.9801, 115.9465, 122.7717),(1.0, 1.0, 1.0), # use for FCOS
    # 'DOTA', (0.3782, 0.3816, 0.3579),(0.1700, 0.1659, 0.1613),
    'DOTA', (0.485, 0.456, 0.406),(0.229, 0.224, 0.225),
    min_size=384, max_size=512)

dota_meta = DatasetMeta(
    'DOTA',
    '/media/pierre/Data_SSD/Datasets/DOTA/prepared/',
    '/media/pierre/Data_SSD/Datasets/DOTA/prepared/categories.json',
    16,
    # 'DOTA', (102.9801, 115.9465, 122.7717),(1.0, 1.0, 1.0), # use for FCOS
    # 'DOTA', (0.3782, 0.3816, 0.3579),(0.1700, 0.1659, 0.1613),
    'DOTA', (0.485, 0.456, 0.406),(0.229, 0.224, 0.225),
    min_size=512, max_size=512,
    prior='/media/pierre/Data_SSD/Datasets/DOTA/classes_prior.json')

dota_test_meta = DatasetMeta(
    'DOTA',
    '/media/pierre/Data_SSD/Datasets/DOTA/val/prepared/',
    '/media/pierre/Data_SSD/Datasets/DOTA/val/prepared/categories.json',
    16,
    'DOTA', (0.485, 0.456, 0.406),(0.229, 0.224, 0.225),
    min_size=512, max_size=512,
    prior='/media/pierre/Data_SSD/Datasets/DOTA/val/prepared/classes_prior.json')

xview_meta = DatasetMeta(
    'DIOR',
    '/home/pierre/Documents/PHD/Datasets/XVIEW/prepared/',
    '/home/pierre/Documents/PHD/Datasets/XVIEW/prepared/categories.json',
    60,
    'DOTA', (0.3782, 0.3816, 0.3579),(0.1700, 0.1659, 0.1613),
    min_size=800, max_size=1333)

dior_meta = DatasetMeta(
    'DIOR',
    '/home/pierre/Documents/PHD/Datasets/DIOR/prepared/train/',
    '/home/pierre/Documents/PHD/Datasets/DIOR/prepared/train/categories.json',
    20,
    'DOTA', (0.3782, 0.3816, 0.3579),(0.1700, 0.1659, 0.1613),
    min_size=800, max_size=800)

mscoco_meta = DatasetMeta(
    'MSCOCO',
    '/home/pierre/Documents/PHD/Datasets/MSCOCO/prepared/train/',
    '/home/pierre/Documents/PHD/Datasets/MSCOCO/prepared/train/categories.json',
    80,
    'DOTA', (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
    min_size=800, max_size=1333)

imagenet_meta = DatasetMeta(
    'IMAGENET',
    '/home/pierre/Documents/PHD/Datasets/ImageNet/ILSVRC/prepared/train',
    '/home/pierre/Documents/PHD/Datasets/ImageNet/ILSVRC/prepared/train/categories.json',
    200,
    'DOTA', (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
    min_size=800, max_size=1333)

