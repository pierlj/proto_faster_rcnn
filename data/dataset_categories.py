import json
import torch
class DatasetCategories():
    def __init__(self, dataset_infos):
        self.categories_path = dataset_infos.categories_path
        with open(self.categories_path) as f:
            self.categories = json.load(f)
        self.names_are_unique = False
        self.reversed_synsets = {v['synset']:k for k,v in self.categories.items()}
        try:
            self.reversed_names = {v['name']: k for k, v in self.categories.items()}
            self.names_are_unique = True
        except:
            pass

    def label_to_name(self, label):
        # TO DO find a better wey to get label given different detection method
        # some have label from 0 to N-1 other from 1 to N.
        label = label + 1 # only for ihm demo 
        if isinstance(label, torch.Tensor):
            label = label.item()
        label = str(label)
        assert label in self.categories.keys(), \
            'Unknown label could be out of range or wrong type'
        return self.categories[label]['name']
    
    def label_to_synset(self, label):
        label = str(label)
        assert label in self.categories.keys(), \
            'Unknown label could be out of range or wrong type'
        return self.categories[label]['synset']
    
    def name_to_label(self, name):
        name = str(name)
        if self.names_are_unique:
            return self.reversed_names[name]
    
    def synset_to_label(self, synset):
        return self.reversed_synset[synset]

