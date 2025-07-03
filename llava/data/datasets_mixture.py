# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)
    ############## ############## ############## ############## ############## ##############
    caption_choice: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    caption_choice_2: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    start_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})
    end_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})


DATASETS_LEGACY = {}


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS_LEGACY:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS_LEGACY.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():
    chartqa = Dataset(
        dataset_name="chartqa",
        dataset_type="torch",
        data_path="/home/qyw/data/chartqa_new/data.json",
        image_path="/home/qyw/data/chartqa_new/images/"
    )
    add_dataset(chartqa)

    ai2d = Dataset(
        dataset_name="ai2d",
        dataset_type="torch",
        data_path="/home/qyw/data/ai2d(internvl)/data.json",
        image_path="/home/qyw/data/ai2d(internvl)/images/"
    )
    add_dataset(ai2d)

    allava_instruct_laion4v = Dataset(
        dataset_name = "allava_instruct_laion4v",
        dataset_type = "torch",
        data_path = "/home/qyw/data/allava_instruct_laion4v/data.json",
        image_path = "/home/qyw/data/allava_instruct_laion4v/images/"
    )
    add_dataset(allava_instruct_laion4v)

    dvqa = Dataset(
        dataset_name = "dvqa",
        dataset_type = "torch",
        data_path = "/home/qyw/data/dvqa/data.json",
        image_path = "/home/qyw/data/dvqa/images/"
    )
    add_dataset(dvqa)

    sharegpt4o = Dataset(
        dataset_name = "sharegpt4o",
        dataset_type = "torch",
        data_path = "/home/qyw/data/sharegpt4o/data.json",
        image_path = "/home/qyw/data/sharegpt4o/images/"
    )
    add_dataset(sharegpt4o)

    textocr = Dataset(
        dataset_name = "textocr",
        dataset_type = "torch",
        data_path = "/home/qyw/data/textocr/data.json",
        image_path = "/home/qyw/data/textocr/images/"
    )
    add_dataset(textocr)

    LLaVA_CC3M_Pretrain_595K = Dataset(
        dataset_name = "LLaVA_CC3M_Pretrain_595K",
        dataset_type = "torch",
        data_path = "../data/LLaVA-CC3M-Pretrain-595K/LLaVA-CC3M-Pretrain-595K/chat.json",
        image_path = "../data/LLaVA-CC3M-Pretrain-595K/LLaVA-CC3M-Pretrain-595K/images/"
    )
    add_dataset(LLaVA_CC3M_Pretrain_595K)

    llava_next = Dataset(
        dataset_name = "llava_next",
        dataset_type = "torch",
        data_path = "/home/qyw/data/LLaVA-NeXT-Data/llava_next_raw_format/llava_next_raw_format_processed.json",
        image_path = "/home/qyw/data/LLaVA-NeXT-Data/llava_next_raw_format/"
    )
    add_dataset(llava_next)

    docvqa = Dataset(
        dataset_name = "docvqa",
        dataset_type = "torch",
        data_path = "/home/qyw/data/docvqa/data.json",
        image_path = "/home/qyw/data/docvqa/images"
    )
    add_dataset(docvqa)

    #debugging(test dataset)
    lmms_chartqa_test = Dataset(
        dataset_name= "chartqa_test",
        dataset_type= "torch",
        data_path = "/home/qyw/VILA/transformed_data.json",
        image_path = "/home/qyw/data/lmms_chartqa/ChartQA/ChartQA Dataset/test/png",
    )
    add_dataset(lmms_chartqa_test)

    lmms_vstar = Dataset(
        dataset_name= "vstar",
        dataset_type= 'torch',
        data_path= "/home/qyw/data/vstar/data_00000.json",
        image_path= "/home/qyw/data/vstar/images",
    )
    add_dataset(lmms_vstar)


