import hashlib
import json
import os

import hydra
import omegaconf
import torch_geometric


def make_hash(o):
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).
    """
    sha1 = hashlib.sha1()
    sha1.update(str.encode(str(o)))
    hash_as_hex = sha1.hexdigest()
    # convert the hex back to int and restrict it to the relevant int range
    seed = int(hash_as_hex, 16) % 4294967295
    return seed


def ensure_serializable(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = ensure_serializable(value)
        return obj
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return {ensure_serializable(item) for item in obj}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, omegaconf.dictconfig.DictConfig):
        return dict(obj)
    else:
        return None


class CustomDataset(torch_geometric.data.Dataset):
    def __init__(self, data_lst):
        super().__init__()
        self.data_lst = data_lst

    def get(self, idx):
        data = self.data_lst[idx]
        keys = list(data.keys())
        return ([data[key] for key in keys], keys)

    def len(self):
        return len(self.data_lst)


class Preprocessor(torch_geometric.data.InMemoryDataset):
    def __init__(self, data_dir, data_list, transforms_config):
        if isinstance(data_list, torch_geometric.data.Dataset):
            data_list = [data_list.get(idx) for idx in range(len(data_list))]
        elif isinstance(data_list, torch_geometric.data.Data):
            data_list = [data_list]
        self.data_list = data_list
        pre_transform = self.instantiate_pre_transform(data_dir, transforms_config)
        super().__init__(self.processed_data_dir, None, pre_transform)
        self.save_transform_parameters()
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return self.root

    @property
    def processed_file_names(self):
        return "data.pt"

    def instantiate_pre_transform(self, data_dir, transforms_config):
        pre_transforms_dict = hydra.utils.instantiate(transforms_config)
        pre_transforms = torch_geometric.transforms.Compose(
            list(pre_transforms_dict.values())
        )
        self.set_processed_data_dir(pre_transforms_dict, data_dir, transforms_config)
        return pre_transforms

    def set_processed_data_dir(self, pre_transforms_dict, data_dir, transforms_config):
        # Use self.transform_parameters to define unique save/load path for each transform parameters
        repo_name = "_".join(list(transforms_config.keys()))
        transforms_parameters = {
            transform_name: transform.parameters
            for transform_name, transform in pre_transforms_dict.items()
        }
        params_hash = make_hash(transforms_parameters)
        self.transforms_parameters = ensure_serializable(transforms_parameters)
        self.processed_data_dir = os.path.join(*[data_dir, repo_name, f"{params_hash}"])

    def save_transform_parameters(self):
        # Check if root/params_dict.json exists, if not, save it
        path_transform_parameters = os.path.join(
            self.processed_data_dir, "path_transform_parameters_dict.json"
        )
        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(self.transforms_parameters, f)
        else:
            # If path_transform_parameters exists, check if the transform_parameters are the same
            with open(path_transform_parameters, "r") as f:
                saved_transform_parameters = json.load(f)

            if saved_transform_parameters != self.transforms_parameters:
                raise ValueError("Different transform parameters for the same data_dir")
            else:
                print(
                    f"Transform parameters are the same, using existing data_dir: {self.processed_data_dir}"
                )

    def process(self):
        self.data_list = [self.pre_transform(d) for d in self.data_list]

        self.data, self.slices = self.collate(self.data_list)
        self._data_list = None  # Reset cache.

        assert isinstance(self._data, torch_geometric.data.Data)
        self.save(self.data_list, self.processed_paths[0])
