from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch


def listdir_nohidden(path):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
    """
    return [f for f in os.listdir(path) if not f.startswith('.')]


def read_office_caltech10_data(dataset_path, domain_name):
    data_paths = []
    data_labels = []
    domain_dir = path.join(dataset_path, domain_name)
    class_names = listdir_nohidden(domain_dir)
    class_names.sort()
    for label, class_name in enumerate(class_names):
        class_dir = path.join(domain_dir, class_name)
        item_names = listdir_nohidden(class_dir)
        for item_name in item_names:
            item_path = path.join(class_dir, item_name)
            data_paths.append(item_path)
            data_labels.append(label)
    return data_paths, data_labels


class OfficeCaltech(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(OfficeCaltech, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_office_caltech10_split_sampler(labels, test_ratio=0.2, num_classes=10):
    """
    :param labels: torch.array(long tensor)
    :param test_ratio: the ratio to split part of the data for test
    :param num_classes: 10
    :return: sampler_train,sampler_test
    """
    sampler_test = []
    sampler_train = []
    for i in range(num_classes):
        loc = torch.nonzero(labels == i)
        loc = loc.view(-1)
        # do random perm to make sure uniform sample
        test_num = round(loc.size(0) * test_ratio)
        loc = loc[torch.randperm(loc.size(0))]
        sampler_test.extend(loc[:test_num].tolist())
        sampler_train.extend(loc[test_num:].tolist())
    sampler_test = SubsetRandomSampler(sampler_test)
    sampler_train = SubsetRandomSampler(sampler_train)
    return sampler_train, sampler_test


def get_office_caltech10_dloader(base_path, domain_name, batch_size, num_workers):
    dataset_path = path.join(base_path, 'dataset', 'OfficeCaltech10')
    data_paths, data_labels = read_office_caltech10_data(dataset_path, domain_name)
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = OfficeCaltech(data_paths, data_labels, transforms_train, domain_name)
    test_dataset = OfficeCaltech(data_paths, data_labels, transforms_test, domain_name)
    sampler_train, sampler_test = get_office_caltech10_split_sampler(torch.LongTensor(data_labels))
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                               sampler=sampler_train)
    test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              sampler=sampler_test)
    return train_dloader, test_dloader
