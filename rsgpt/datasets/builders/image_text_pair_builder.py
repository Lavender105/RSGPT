import os
import logging
import warnings

from rsgpt.common.registry import registry
from rsgpt.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from rsgpt.datasets.datasets.cc_sbu_dataset import CCSBUAlignDataset
from rsgpt.datasets.datasets.rsicd_dataset import RSICDDataset
from rsgpt.datasets.datasets.rsicd_instruction_dataset import RSICDInstructionDataset
from rsgpt.datasets.datasets.rsicap_instruction_dataset import RSICapInstructionDataset
from rsgpt.datasets.datasets.rsvqahr_instruction_dataset import RSVQAHRInstructionDataset
from rsgpt.datasets.datasets.rsvqalr_instruction_dataset import RSVQALRInstructionDataset
from rsgpt.datasets.datasets.ucm_instruction_dataset import UCMInstructionDataset
from rsgpt.datasets.datasets.sydney_instruction_dataset import SydneyInstructionDataset


@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets

@registry.register_builder("rsicd")
class RSICDBuilder(BaseDatasetBuilder):
    train_dataset_cls = RSICDDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rsicd/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))
        # import pdb; pdb.set_trace()
        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'rsicd_cap_processed_summary.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets

@registry.register_builder("rsicd_instruction")
class RSICDInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = RSICDInstructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rsicd_instruction/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))
        # import pdb; pdb.set_trace()
        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'rsicd_cap_processed_instruction_train.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets

@registry.register_builder("rsicap_instruction")
class RSICapInstructionDataset(BaseDatasetBuilder):
    train_dataset_cls = RSICapInstructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rsicap_instruction/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))
        # import pdb; pdb.set_trace()
        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'captions.json')],
            vis_root=os.path.join(storage_path, 'images'),
        )

        return datasets


@registry.register_builder("rsvqahr_instruction")
class RSVQAHRInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = RSVQAHRInstructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rsvqahr_instruction/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))
        # import pdb; pdb.set_trace()
        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'rsvqa_hr_train.json')],
            vis_root=os.path.join(storage_path, 'Data'),
        )

        return datasets

@registry.register_builder("rsvqalr_instruction")
class RSVQALRInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = RSVQALRInstructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rsvqalr_instruction/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))
        # import pdb; pdb.set_trace()
        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'rsvqa_lr_train.json')],
            vis_root=os.path.join(storage_path, 'Images_LR'),
        )

        return datasets

@registry.register_builder("ucm_instruction")
class UCMInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = UCMInstructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/ucm_instruction/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))
        # import pdb; pdb.set_trace()
        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'ucm_cap_processed_instruction_train.json')],
            vis_root=os.path.join(storage_path, 'imgs'),
        )

        return datasets


@registry.register_builder("sydney_instruction")
class SydneyInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = SydneyInstructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sydney_instruction/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))
        # import pdb; pdb.set_trace()
        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'sydney_cap_processed_instruction_train.json')],
            vis_root=os.path.join(storage_path, 'imgs'),
        )

        return datasets


