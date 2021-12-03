from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch


def read_mini_domainnet_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits_mini", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            data_paths.append(data_path)
            data_labels.append(label)
    return data_paths, data_labels


class MiniDomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(MiniDomainNet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_mini_domainnet_dloader(base_path, domain_name, batch_size, num_workers):
    dataset_path = path.join(base_path, 'dataset', 'DomainNet')
    train_data_paths, train_data_labels = read_mini_domainnet_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_mini_domainnet_data(dataset_path, domain_name, split="test")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])

    train_dataset = MiniDomainNet(train_data_paths, train_data_labels, transforms_train, domain_name)
    test_dataset = MiniDomainNet(test_data_paths, test_data_labels, transforms_test, domain_name)
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                               shuffle=True)
    test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              shuffle=True)
    return train_dloader, test_dloader
