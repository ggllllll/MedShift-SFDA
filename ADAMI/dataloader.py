#!/usr/env/bin python3.6

import io
import re
import random
from operator import itemgetter
from pathlib import Path
from itertools import repeat
from functools import partial
from typing import Any, Callable, BinaryIO, Dict, List, Match, Pattern, Tuple, Union
import csv
from multiprocessing import cpu_count
import torch
import numpy as np
from torch import Tensor
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from utils import id_, map_, class2one_hot, augment, read_nii_image,read_unknownformat_image
from utils import simplex, sset, one_hot, pad_to, remap

F = Union[Path, BinaryIO]
D = Union[Image.Image, np.ndarray, Tensor]

def convert_labeled_list(csv_list, r=1):
    img_pair_list = list()
    for csv_file in csv_list:
        with open(csv_file, 'r') as f:
            img_in_csv = f.read().split('\n')[1:-1]
        img_pair_list += img_in_csv
    img_list = [i.split(',')[0] for i in img_pair_list]
    if len(img_pair_list[0].split(',')) == 1:
        label_list = None
    else:
        label_list = [i.split(',')[-1].replace('.tif', '-{}.tif'.format(r)) for i in img_pair_list]
    return img_list, label_list


class Sampler(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
def get_loaders(args, data_folder: str, subfolders:str,
                batch_size: int, n_class: int,
                debug: bool, in_memory: bool, dtype, shuffle:bool, mode:str, val_subfolders:"") -> Tuple[DataLoader, DataLoader]:

    nii_transform2 = transforms.Compose([
        lambda nd: torch.tensor(nd, dtype=torch.float32),
        lambda nd: nd[:,0:384,0:384],
    ])

    nii_gt_transform2 = transforms.Compose([
        lambda nd: torch.tensor(nd, dtype=torch.int64),
        partial(class2one_hot, C=n_class),
        lambda nd: nd[:,:,0:384,0:384],
        itemgetter(0),
    ])


    nii_transform = transforms.Compose([
        lambda nd: torch.tensor(nd, dtype=torch.float32),
        lambda nd: (nd+4) / 8.5,  # max <= 1
    ])

    nii_gt_transform = transforms.Compose([
        lambda nd: torch.tensor(nd, dtype=torch.int64),
        partial(class2one_hot, C=n_class),
        itemgetter(0),
    ])

    png_transform = transforms.Compose([
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=dtype)
    ])
    imnpy_transform = transforms.Compose([
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=dtype)
    ])
    npy_transform = transforms.Compose([
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=dtype)
    ])
    gtnpy_transform = transforms.Compose([
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: torch.tensor(nd, dtype=torch.int64),
        partial(class2one_hot, C=n_class),
        itemgetter(0)
    ])
    gt_transform = transforms.Compose([
        lambda nd: torch.tensor(nd, dtype=torch.int64),
        partial(class2one_hot, C=n_class),
        itemgetter(0),
    ])
    gtpng_transform = transforms.Compose([
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: torch.tensor(nd, dtype=torch.int64),
        partial(class2one_hot, C=n_class),
        itemgetter(0),
    ])


    if mode == "target":
        losses = eval(args.target_losses)
    else:
        losses = eval(args.source_losses)

    bounds_generators: List[Callable] = []
    for _, _, bounds_name, bounds_params, fn, _ in losses:
        if bounds_name is None:
            bounds_generators.append(lambda *a: torch.zeros(n_class, 1, 2))
            continue
        bounds_class = getattr(__import__('bounds'), bounds_name)
        bounds_generators.append(bounds_class(C=args.n_class, fn=fn, **bounds_params))

    folders_list = eval(subfolders)
    val_folders_list = eval(subfolders)
    if val_subfolders !="":
        val_folders_list = eval(val_subfolders)

    # print(folders_list)
    folders, trans, are_hots = zip(*folders_list)
    valfolders, val_trans, val_are_hots = zip(*val_folders_list)
    # Create partial functions: Easier for readability later (see the difference between train and validation)
    gen_dataset = partial(SliceDataset,
                          transforms=trans,
                          are_hots=are_hots,
                          debug=debug,
                          C=n_class,
                          in_memory=in_memory, augment=args.augment,
                          bounds_generators=bounds_generators)
    valgen_dataset = partial(SliceDataset,
                          transforms=val_trans,
                          are_hots=val_are_hots,
                          debug=debug,
                          C=n_class,
                          in_memory=in_memory, augment=args.augment,
                          bounds_generators=bounds_generators)

    data_loader = partial(DataLoader,
                          num_workers=0,
                          #num_workers=min(cpu_count(), batch_size + 4),
                          #num_workers=1,
                          pin_memory=True)

    # Prepare the datasets and dataloaders
    tr_csv = tuple(args.target_dataset)
    ts_csv = tuple(args.test_dataset)
    tr_img_list, tr_label_list = convert_labeled_list(tr_csv, r=1)
    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=1)

    # train_folders: List[Path] = [Path(data_folder, "train", f) for f in folders]
    # if args.trainval:
    #     train_folders: List[Path] = [Path(data_folder, "trainval", f) for f in folders]
    # elif args.valonly:
    #     train_folders: List[Path] = [Path(data_folder, "val", f) for f in folders]
    # I assume all files have the same name inside their folder: makes things much easier
    # train_names: List[str] = map_(lambda p: str(p.name), train_folders[0].glob("*.png"))
    # if len(train_names)==0:
    #     train_names: List[str] = map_(lambda p: str(p.name), train_folders[0].glob("*.nii"))
    # if len(train_names)==0:
    #     train_names: List[str] = map_(lambda p: str(p.name), train_folders[0].glob("*.npy"))
    
    train_set = gen_dataset(args.root, tr_img_list,
                            tr_label_list)
    train_loader = data_loader(train_set,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               drop_last=False)

    # if args.ontest:
    #     print('Inference will be done on "test" folder')
    #     val_folders: List[Path] = [Path(data_folder, "test", f) for f in valfolders]
    # elif args.ontrain:
    #     print('Inference will be done on "train" folder')
    #     val_folders: List[Path] = [Path(data_folder, "train", f) for f in valfolders]
    # else:#/
    #     print('Inference will be done on "val" folder')
    #     val_folders: List[Path] = [Path(data_folder, "val", f) for f in valfolders]
    # val_names: List[str] = map_(lambda p: str(p.name), val_folders[0].glob("*.png"))
    # if len(val_names)==0:
    #     val_names: List[str] = map_(lambda p: str(p.name), val_folders[0].glob("*.nii"))
    # if len(val_names)==0:
    #     val_names: List[str] = map_(lambda p: str(p.name), val_folders[0].glob("*.npy"))

    val_set = valgen_dataset(args.root, ts_img_list,
                          ts_label_list)


    val_loader = data_loader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False)

    return train_loader, val_loader



class SliceDataset(Dataset):
    def __init__(self, root, filenames_image: List[str], filenames_mask: List[str], are_hots: List[bool],
                 bounds_generators: List[Callable], transforms: List[Callable], debug=False,  augment: bool = False,
                 C=2, in_memory: bool = False) -> None:
        self.root = root
        self.file_image: List[str] = filenames_image
        self.file_mask: List[str] = filenames_mask
        self.transforms: List[Callable[[D], Tensor]] = transforms

        self.are_hots: List[bool] = are_hots
        self.debug = debug
        self.C: int = C  # Number of classes
        self.in_memory: bool = in_memory
        self.bounds_generators: List[Callable] = bounds_generators
        self.augment: bool = augment

        # if self.debug:
        #     self.filenames = self.filenames[:10]
        
        # assert self.check_files()  # Make sure all file exists

        # Load things in memory if needed
        # self.files: List[List[F]] = SliceDataset.load_images(self.folders, self.filenames, self.in_memory)
        # assert len(self.files) == len(self.folders)
        # for files in self.files:
        #     assert len(files) == len(self.filenames)

        print(f"Initialized {self.__class__.__name__} with {len(self.file_image)} images")

    def check_files(self) -> bool:
        for folder in self.folders:
            #print(folder)
            if not Path(folder).exists():
                print(folder, "does not exist")
                return False

            for f_n in self.filenames:
                #print(f_n)
                if not Path(folder, f_n).exists():
                    print(folder,f_n, "does not exist")
                    return False

        return True

    @staticmethod
    def load_images(folders: List[Path], filenames: List[str], in_memory: bool) -> List[List[F]]:
        def load(folder: Path, filename: str) -> F:
            p: Path = Path(folder, filename)
            if in_memory:
                with open(p, 'rb') as data:
                    res = io.BytesIO(data.read())
                return res
            return p
        if in_memory:
            print("Loading the data in memory...")

        files: List[List[F]] = [[load(f, im) for im in filenames] for f in folders]

        return files

    def __len__(self):
        return len(self.file_image)

    def __getitem__(self, index: int) -> List[Any]:
        img_file = os.path.join(self.root, self.file_image[index])
        img = Image.open(img_file)
        img = img.resize((512, 512))
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        bounds = [f(self.file_image[index]) for f in self.bounds_generators]

        if self.file_mask != None:
            label_file = os.path.join(self.root, self.file_mask[index])
            label = Image.open(label_file)
            label = label.resize((512, 512), resample=Image.NEAREST)
            label_npy = np.array(label)
            mask = np.zeros_like(label_npy)
            mask[label_npy > 0] = 1
            mask[label_npy == 128] = 2
            mask = mask[np.newaxis]
            return img_file, img_npy, mask, bounds

        return img_file, img_npy, '', bounds




class PatientSampler(Sampler):
    def __init__(self, dataset: SliceDataset, grp_regex, shuffle=False) -> None:
        filenames: List[str] = dataset.filenames
        # Might be needed in case of escape sequence fuckups
        # self.grp_regex = bytes(grp_regex, "utf-8").decode('unicode_escape')
        self.grp_regex = grp_regex

        # Configure the shuffling function
        self.shuffle: bool = shuffle
        self.shuffle_fn: Callable = (lambda x: random.sample(x, len(x))) if self.shuffle else id_

        print(f"Grouping using {self.grp_regex} regex")
        # assert grp_regex == "(patient\d+_\d+)_\d+"
        # grouping_regex: Pattern = re.compile("grp_regex")
        grouping_regex: Pattern = re.compile(self.grp_regex)

        stems: List[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
        matches: List[Match] = map_(grouping_regex.match, stems)
        patients: List[str] = [match.group(0) for match in matches]

        unique_patients: List[str] = list(set(patients))
        assert len(unique_patients) <= len(filenames)
        print(f"Found {len(unique_patients)} unique patients out of {len(filenames)} images")

        self.idx_map: Dict[str, List[int]] = dict(zip(unique_patients, repeat(None)))
        for i, patient in enumerate(patients):
            if not self.idx_map[patient]:
                self.idx_map[patient] = []

            self.idx_map[patient] += [i]
        assert sum(len(self.idx_map[k]) for k in unique_patients) == len(filenames)

        print("Patient to slices mapping done")

    def __len__(self):
        return len(self.idx_map.keys())

    def __iter__(self):
        values = list(self.idx_map.values())
        shuffled = self.shuffle_fn(values)
        return iter(shuffled)


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples




class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class Concat(Dataset):

    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length
