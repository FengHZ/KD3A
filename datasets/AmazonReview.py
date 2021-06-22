import numpy as np
import time
import pickle
from scipy.sparse import coo_matrix
import torch.utils.data as data
from PIL import Image
import numpy as np
from scipy.io import loadmat
from os import path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class AmazonReviewDataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        tensor, label = np.squeeze(np.asarray(self.data[index])), self.labels[index]
        return tensor, label

    def __len__(self):
        return len(self.data)


def load_amazon(base_path):
    dimension = 5000
    amazon = np.load(path.join(base_path, "dataset", "AmazonReview", "amazon.npz"))
    amazon_xx = coo_matrix((amazon['xx_data'], (amazon['xx_col'], amazon['xx_row'])),
                           shape=amazon['xx_shape'][::-1]).tocsc()
    amazon_xx = amazon_xx[:, :dimension]
    amazon_yy = amazon['yy']
    amazon_yy = (amazon_yy + 1) / 2
    amazon_offset = amazon['offset'].flatten()
    # Partition the data into four categories and for each category partition the data set into training and test set.
    data_name = ["books", "dvd", "electronics", "kitchen"]
    num_data_sets = 4
    data_insts, data_labels, num_insts = [], [], []
    for i in range(num_data_sets):
        data_insts.append(amazon_xx[amazon_offset[i]: amazon_offset[i + 1], :])
        data_labels.append(amazon_yy[amazon_offset[i]: amazon_offset[i + 1], :])
        num_insts.append(amazon_offset[i + 1] - amazon_offset[i])
        # Randomly shuffle.
        r_order = np.arange(num_insts[i])
        np.random.shuffle(r_order)
        data_insts[i] = data_insts[i][r_order, :]
        data_labels[i] = data_labels[i][r_order, :]
        data_insts[i] = data_insts[i].todense().astype(np.float32)
        data_labels[i] = data_labels[i].ravel().astype(np.int64)
    return data_insts, data_labels


def amazon_dataset_read(base_path, domain, batch_size):
    data_insts, data_labels = load_amazon(base_path)
    if domain == "books":
        train_image, train_label, test_image, test_label = data_insts[0][:2000], data_labels[0][:2000], data_insts[0][
                                                                                                        2000:], \
                                                           data_labels[0][2000:]
    elif domain == "dvd":
        train_image, train_label, test_image, test_label = data_insts[1][:2000], data_labels[1][:2000], data_insts[1][
                                                                                                        2000:], \
                                                           data_labels[1][2000:]
    elif domain == "electronics":
        train_image, train_label, test_image, test_label = data_insts[2][:2000], data_labels[2][:2000], data_insts[2][
                                                                                                        2000:], \
                                                           data_labels[2][2000:]
    elif domain == "kitchen":
        train_image, train_label, test_image, test_label = data_insts[3][:2000], data_labels[3][:2000], data_insts[3][
                                                                                                        2000:], \
                                                           data_labels[3][2000:]
    else:
        raise NotImplementedError("Domain {} Not Implemented".format(domain))
    # raise train and test data loader
    train_dataset = AmazonReviewDataset(data=train_image, labels=train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = AmazonReviewDataset(data=test_image, labels=test_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader
