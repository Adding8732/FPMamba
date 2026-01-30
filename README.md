# Frequency-Guided State Space Model for Enhanced Remote Sensing Image Dehazing

[Xinxin Miao],[Xiyuan Wang],[Bo Wang]

## Update
- 2026.01.29 	:Release the model and code

>Remote sensing images are often degraded by haze, leading to reduced contrast and visibility, which hinders downstream vision tasks. Traditional dehazing methods rely heavily on a priori knowledge, limiting their effectiveness in complex scenes. Recent deep learning approaches, particularly those based on convolutional neural networks (CNNs) and Transformers, have shown promise but are constrained by computational overhead and limited frequency domain utilization. Inspired by state space models (SSMs), we propose a Frequency-Prompted State Space Model (FPMamba) that integrates spatial and frequency domain features for effective dehazing. Our model introduces a Frequency-Decoupled Prompting Block to extract and modulate frequency components, guiding the dehazing process. Extensive experiments on publicly available datasets demonstrate that FPMamba outperforms several state-of-the-art methods, achieving superior dehazing performance with improved detail preservation and structural integrity.

# Installation

This repository is built in PyTorch 2.0.0 and tested on Ubuntu 22.04 environment (Python3.10, CUDA11.8).
Follow these intructions

1.Make conda environment
```
conda create -n pytorch2 python=3.10
conda activate pytorch2
```

2.Install dependencies
```
conda install pytorch=2.0.0 torchvision cudatoolkit=11.8 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```
3.Previous installation To use the selective scan with efficient hard-ware design, the `mamba_ssm` library is needed to install with the folllowing command.
```
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```

4.Install basicsr
```
python setup.py develop --no_cuda_ext
```

## Training and Evaluation

To train FPM on StateHaze1K、LHID、DHID and RICE, you can run:
```
The pre-trained models of this work have been made publicly available for testing. The full source code will be released upon the acceptance of this paper by The Visual Computer. The manuscript is currently under submission to this journal.
```

To evaluate FPM, you can refer commands in 'test.sh'

## Download the Datasets and Results

| Datasets    | Download the Datasets                                             | Trained Models                                                    |
|-------------|-------------------------------------------------------------------|-------------------------------------------------------------------|
| StateHaze1k | [Baidu](https://pan.baidu.com/s/1yeSeSo0HcA1zN-DRfrJz8w?pwd=0n4s) | [Baidu](https://pan.baidu.com/s/10rnPdoBVJa8oI26QNMDjPg?pwd=0n4s) |
| LHID        | [Baidu](https://pan.baidu.com/s/1FAtCGYenF0Qcq_jVYu2nyg?pwd=0n4s) | [Baidu](https://pan.baidu.com/s/10rnPdoBVJa8oI26QNMDjPg?pwd=0n4s) |
| DHID        | [Baidu](https://pan.baidu.com/s/1FAtCGYenF0Qcq_jVYu2nyg?pwd=0n4s) | [Baidu](https://pan.baidu.com/s/10rnPdoBVJa8oI26QNMDjPg?pwd=0n4s) |
| RICE        | [Baidu](https://pan.baidu.com/s/1cWuq68BfXHPhyqnh-0BGtw?pwd=0n4s) | [Baidu](https://pan.baidu.com/s/10rnPdoBVJa8oI26QNMDjPg?pwd=0n4s) |
| Dense-Haze  | [Baidu](https://pan.baidu.com/s/1ZTmAkU4Tx5C0z17-Yehx6Q?pwd=0n4s) | [Baidu](https://pan.baidu.com/s/10rnPdoBVJa8oI26QNMDjPg?pwd=0n4s) |
| NH-Haze     | [Baidu](https://pan.baidu.com/s/1ZTmAkU4Tx5C0z17-Yehx6Q?pwd=0n4s) | [Baidu](https://pan.baidu.com/s/10rnPdoBVJa8oI26QNMDjPg?pwd=0n4s) |


## Citation
If you find this project useful, please consider citing:

    @inproceedings{FPM,
      title={Frequency-Guided State Space Model for Enhanced Remote Sensing Image Dehazing},
      author={Xinxin Miao, Xiyuan Wang, Bo Wang},
      year={2026}
    }

## Acknowledgement

This code borrows heavily from [MambaIR](https://github.com/csguoh/MambaIR). 
