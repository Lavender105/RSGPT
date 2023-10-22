
<font size='5'>**RSGPT: A Remote Sensing Vision Language Model and Benchmark**</font>

Yuan Hu, Jianlong Yuan, Congcong Wen, Xiaonan Lu, Xiang Li☨

☨corresponding author

<!-- <a href='https://rsgpt.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  -->
<a href='https://arxiv.org/abs/2307.15266'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>


## :fire: Updates
* **[2023.10.22]** We release the evaluation dataset RSIEval. Please fill this [form](https://docs.google.com/forms/d/1h5ydiswunM_EMfZZtyJjNiTMpeOzRwooXh73AOqokzU/edit) to get the dataset.

## Dataset
* RSICap: 2,585 image-text pairs with high-quality human-annotated captions.
* RSIEval: 100 high-quality human-annotated captions with 936 open-ended visual question-answer pairs.

## Code
The idea of finetuning our vision-language model is borrowd from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4).
Our model is based on finetuning [InstructBLIP](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md) using our RSICap dataset.

## Acknowledgement
+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). A pupular open-source vision-language model.
+ [InstructBLIP](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md). The model architecture of RSGPT follows InstructBLIP. Don't forget to check this great open-source work if you don't know it before!
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

