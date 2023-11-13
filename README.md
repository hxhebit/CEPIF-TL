# A Cross Environment Person Identification Framework Based on Transfer Learning Using IR-UWB Radar

PyTorch implementation of the paper: A Cross Environment Person Identification Framework Based on Transfer Learning Using IR-UWB Radar

## About CEPIF-TL

We propose a transfer learning framework for achieving cross-environment person identification. A neural network is devised for  mapping signals from distinct environments into a unified feature space and further align them, enabling the model to extract environment-independent features. 

## Getting Started

### Dependencies

```python
pytorch==1.12.1
tensorboard==2.10.1
torchvision==0.13.1
configargparse==1.4
numpy==1.21.5
scikit_learn==1.1.2
timm==0.5.4 
```

### Dataset

Using a real Impulse Radio Ultra-WideBand (IR-UWB) radar testbed, we build a dataset with 22,264 samples from three environments, varying in testing distance and occlusion condition. The directory structure is:

```
│path/to/dataset/
├──A/
│  ├── p1-A-0.5m-F-1
│  │   ├── 1.png
│  │   ├── 1_mw.png
│  │   ├── 1_pt.png
│  │   ├── ......
│  ├── ......
├──A_valid/
│  ├── p1-A-0.5m-F-1
│  │   ├── 2.png
│  │   ├── 3.png
│  │   ├── ......
│  ├── ......
├──A_test/
│  ├── p1-A-0.5m-F-1
│  │   ├── 4.png
│  │   ├── 5.png
│  │   ├── ......
│  ├── ......
```
It is worth noting that the initial version of our work provides a part of dataset , while the remaining dataset will be made available in subsequent versions.


### Usage

Frist, clone the repository locally:

```bash
git clone https://github.com/hxhebit/CEPIF-TL.git
```

Then, install Pytorch, tensorboard, and other dependencies:

```
pip3 install -r requirements.txt
```

Next, configure parameters at [file](https://github.com/hxhebit/CEPIF-TL.git/run/args/task1.yaml).

Finally, to train and test on a single node with 8 GPUs, run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --master_port 29500 main.py --config run/args/task1.yaml --data_dir ./dataset --src_domain A_train --tgt_domain C_train --tgt_domain_valid C_valid --tgt_domain_test C_test --save_folder task1
```

To customize your dataset, simply substitute the [dataset](https://github.com/hxhebit/CEPIF-TL.git/dataset) with an identical directory structure.

---

***This project is continuously being updated.*** 