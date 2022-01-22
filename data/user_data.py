import os
import numpy as np
import torch


class RegistrationData:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.template = self.find_attribute('template')
        self.source = self.find_attribute('source')
        self.transformation = self.find_attribute('transformation')
        self.check_data()

    def find_attribute(self, attribute):
        try:
            attribute_data = self.data[attribute]
        except:
            print("Given data directory has no key attribute \"{}\"".format(attribute))
        return attribute_data

    def check_data(self):
        assert 1 < len(self.template.shape) < 4, "Error in dimension of point clouds! Given data dimension: {}".format(self.template.shape)
        assert 1 < len(self.source.shape) < 4, "Error in dimension of point clouds! Given data dimension: {}".format(self.source.shape)
        assert 1 < len(self.transformation.shape) < 4, "Error in dimension of transformations! Given data dimension: {}".format(self.transformation.shape)

        if len(self.template.shape) == 2:
            self.template = self.template.reshape(1, -1, 3)
        if len(self.source.shape) == 2:
            self.source = self.source.reshape(1, -1, 3)
        if len(self.transformation.shape) == 2:
            self.transformation = self.transformation.reshape(1, 4, 4)

        assert self.template.shape[0] == self.source.shape[0], "Inconsistency in the number of template and source point clouds!"
        assert self.source.shape[0] == self.transformation.shape[0], "Inconsistency in the number of transformation and source point clouds!"

    def __len__(self):
        return self.template.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.template[index]).float(), torch.tensor(self.source[index]).float(), torch.tensor(self.transformation[index]).float()
