
<font size='5'>**RSGPT: A Remote Sensing Vision Language Model and Benchmark**</font>

[Yuan Hu](https://scholar.google.com.sg/citations?user=NFRuz4kAAAAJ&hl=zh-CN), Jianlong Yuan, Congcong Wen, Xiaonan Lu, [Xiang Li☨](https://xiangli.ac.cn)

☨corresponding author

<!-- <a href='https://rsgpt.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  -->
<a href='https://arxiv.org/abs/2307.15266'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

This is an ongoing project. We are working on increasing the dataset size.

## Related Projects

<font size='5'>**VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding**</font>

Xiang Li, Jian Ding, Mohamed Elhoseiny

<a href='https://vrsbench.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2406.12384'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>  <a href='https://huggingface.co/datasets/xiang709/VRSBench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'>

<font size='5'>**Vision-language models in remote sensing: Current progress and future trends**</font>

[Xiang Li☨](https://xiangli.ac.cn), [Congcong Wen](https://wencc.xyz/), [Yuan Hu](https://scholar.google.com.sg/citations?user=NFRuz4kAAAAJ&hl=zh-CN), Zhenghang Yuan, [Xiao Xiang Zhu](https://www.professoren.tum.de/en/zhu-xiaoxiang)

<a href='[https://arxiv.org/abs/2307.15266](https://ieeexplore.ieee.org/abstract/document/10506064/)'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>


## :fire: Updates
* **[2025.05.08]** We release the code for training and testing RSGPT.
* **[2024.12.18]** We release the [manual scoring results](https://drive.google.com/file/d/1e3joLIiWfUgena17Dx8wZPWGNjs7vGua/view?usp=sharing) for RSIEval.
* **[2024.06.19]** We release the VRSBench, A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding. VRSBench contains 29,614 images, with 29,614 human-verified detailed captions, 52,472 object references, and 123,221 question-answer pairs. check [VRSBench Project Page](https://vrsbench.github.io/).
* **[2024.05.23]** We release the RSICap dataset. Please fill out this [form](https://docs.google.com/forms/d/1h5ydiswunM_EMfZZtyJjNiTMpeOzRwooXh73AOqokzU/edit) to get both RSICap and RSIEval dataset.
* **[2023.11.10]** Our survey about vision-language models in remote sensing. [RSVLM](https://arxiv.org/pdf/2305.05726.pdf).
* **[2023.10.22]** The RSICap dataset and code will be released upon paper acceptance.
* **[2023.10.22]** We release the evaluation dataset RSIEval. Please fill out this [form](https://docs.google.com/forms/d/1h5ydiswunM_EMfZZtyJjNiTMpeOzRwooXh73AOqokzU/edit) to get both the RSIEval dataset.

## Dataset
* RSICap: 2,585 image-text pairs with high-quality human-annotated captions.
* RSIEval: 100 high-quality human-annotated captions with 936 open-ended visual question-answer pairs.

## Code
The idea of finetuning our vision-language model is borrowed from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4).
Our model is based on finetuning [InstructBLIP](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md) using our RSICap dataset.

## 🚀 Installation
Set up a conda environment using the provided `environment.yml` file:

# Step 1: Create the environment
```
conda env create -f environment.yml
```

# Step 2: Activate the environment
```
conda activate rsgpt
```

## Training
```
torchrun --nproc_per_node=8 train.py --cfg-path train_configs/rsgpt_train.yaml
```

## Testing
Test image captioning:
```
python test.py --cfg-path eval_configs/rsgpt_eval.yaml --gpu-id 0 --out-path rsgpt/output --task ic
```

Test visual question answering:
```
python test.py --cfg-path eval_configs/rsgpt_eval.yaml --gpu-id 0 --out-path rsgpt/output --task vqa
```

## Acknowledgement
+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). A popular open-source vision-language model.
+ [InstructBLIP](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md). The model architecture of RSGPT follows InstructBLIP. Don't forget to check out this great open-source work if you don't know it before!
+ [Lavis](https://github.com/salesforce/LAVIS). This repository is built upon Lavis!
+ [Vicuna](https://github.com/lm-sys/FastChat). The fantastic language ability of Vicuna with only 13B parameters is just amazing. And it is open-source!


If you're using RSGPT in your research or applications, please cite using this BibTeX:

```bibtex
@article{hu2023rsgpt,
  title={RSGPT: A Remote Sensing Vision Language Model and Benchmark},
  author={Hu, Yuan and Yuan, Jianlong and Wen, Congcong and Lu, Xiaonan and Li, Xiang},
  journal={arXiv preprint arXiv:2307.15266},
  year={2023}
}
```

