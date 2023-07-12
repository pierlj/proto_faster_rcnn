from .dataset import ObjectDetectionDataset

dataset = ObjectDetectionDataset(
    'DOTA', '/home/pierre/Documents/Datasets/DOTA/prepared/')

loader = dataset.get_dataloader()

for i, batch in enumerate(loader):
    if i == 100:
        break
