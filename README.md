# VMoBA: Mixture-of-Block Attention for Video Diffusion Models

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2410.08261-b31b1b.svg)](TODO)

</div>

![teaser](assets/images/teaser.png)

## 🚀 TL;DR

We introduce **VMoBA**, Mixture of Block Attention for Video Diffusion Models!

- 🌟 Sparse attention mechanism based on MoBA, designed for video diffusion model **training**.
- 🖼️ Key innovations: Layer-wise Recurrent Block Partition, Global Block Selection, and Threshold-based Block Selection. These innovations make VMoBA performance better and quicker in video generation.
- ✨ 2.92x FLOPs acceleration. 1.48x latency acceleration on 576p video (93x576x1024, 55K tokens). Faster with longer sequence length!

![](assets/images/architecture.png)


## 🎉 News

- [2025-6-27] The code of VMoBA is released!

## 🛠️ Quick Start

We provide a **clean single-file code** with only VMoBA implmented by FlashAttention and its speed test unit. Feel free to replace Full Attention with VMoBA in any of your models!

### Environment Preparation

``` bash
# Create a new environment with Conda
conda create -n diffusers python=3.11
conda activate diffusers

# Install Pytorch
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

## Install FlashAttention locally
pip install packaging ninja
mkdir libs
cd libs
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.6.3+cu123torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

## Install other dependencies
pip install -r requirements.txt
```

For issues installing FlashAttention, please refer to the [official repo](https://github.com/Dao-AILab/flash-attention) for help.


### VMoBA Speed Test

VMoBA is be implemented in single file, `src/vmoba.py`

Run this command to test the speed compared with Full Attention.

``` python
CUDA_VISIBLE_DEVICES=1 \
python -u src/vmoba.py
```

Feel free to try different sequence length and component variables (topk selection, local selection as in the vanilla MoBA).

Note: The current implementation based on FlashAttention shows clear acceleration than Full Attention when the sequence length being larger than roughly 33K Tokens. This is also suggested by [one of MoBA's issue](https://github.com/MoonshotAI/MoBA/issues/9).

Note2: The 1-2-3D block partition algorithm is implmented in `process_moba_input` and `process_moba_output` functions in the same file. Please use it according to your data format. 


### Theoretic FLOPs computation

In case that most third-party packages to compute FLOPs of attention-based networks usually miss some operators, (Lack of implementation for certain operators.), we implement a hand-draft theoretic FLOPs computation script to compute the theoretic FLOPs of VMoBA and Full Attention networks. The code is at `src/cal_theo_flops.py`.

``` python
python scripts/flops/cal_theo_flops.py
```




## Citation

```
article{wu2025vmoba,
  title={VMoBA: Mixture-of-Block Attention for Video Diffusion Models},
  author={Jianzong Wu, Liang Hou, Haotian Yang, Xin Tao, Ye Tian, Pengfei Wan, Di Zhang, and Yunhai Tong},
  journal={arXiv preprint arXiv:TODO},
  year={2025},
}
```


<p align="center">
  <a href="https://star-history.com/#KwaiVGI/VMoBA&Date">
    <img src="https://api.star-history.com/svg?repos=KwaiVGI/VMoBA&type=Date" alt="Star History Chart">
  </a>
</p>