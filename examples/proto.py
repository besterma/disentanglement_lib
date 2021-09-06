import os
os.environ["DISENTANGLEMENT_LIB_DATA"] = "/home/besterma/ETH/Semester_Thesis/Python/disentanglement_lib/data/"
from disentanglement_lib.data.ground_truth import util
from disentanglement_lib.data.ground_truth import named_data

from torch.utils.data import DataLoader



dataset = named_data.get_named_ground_truth_data("mpi3d_toy")
train_loader = DataLoader(util.torch_data_set_generator_from_ground_truth_data(dataset, 7), num_workers=0)
print(next(iter(train_loader)))
print(len(train_loader.dataset))
