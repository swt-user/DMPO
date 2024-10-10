# Direct Multi-Turn Preference Optimization for Language Agent

<img src="https://github.com/yuanmengqi/DMPO/blob/main/DMPO_structure.png" alt="Image" width="700"/>

This repository contains the official code for our paper [Direct Multi-Turn Preference Optimization for Language Agents](https://arxiv.org/abs/2406.14868).  (EMNLP 2024 Main Conference)

## Setup
You can set up the environment and download the data by running `bash setup.sh`.

## Run
You can complete the DMPO pipeline by running `run_dmpo.sh <DATASET> <BASIC_MODEL_PATH> <NEW_MODEL_SAVING_PATH>`. The script contains three sections:
* Training and evaluating the SFT model
* Constructing the DMPO training dataset
* Training and evaluating the DMPO model
  
Similarly, you can run the code `run_dmpo_mistral.sh <DATASET> <BASIC_MODEL_PATH> <NEW_MODEL_SAVING_PATH>` to perform training using the Mistral model.
## Citation
If you find this code useful, please cite our paper:
```
@misc{shi2024directmultiturnpreferenceoptimization,
      title={Direct Multi-Turn Preference Optimization for Language Agents}, 
      author={Wentao Shi and Mengqi Yuan and Junkang Wu and Qifan Wang and Fuli Feng},
      year={2024},
      eprint={2406.14868},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.14868}, 
}
```