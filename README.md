## TAL - Official PyTorch Implementation
![image](https://github.com/user-attachments/assets/e0f6294a-54f8-45b9-8a29-d78865df930a)


### [NeurIPS 2024] Typicalness-Aware Learning for Failure Detection
Yijun Liu, Jiequan Cui, Zhuotao Tian*, Senqiao Yang, Qingdong He, Xiaoling Wang, and Jingyong Su*<br>

[Paper](https://arxiv.org/abs/2411.01981)

### Abstract
Deep neural networks (DNNs) often suffer from the overconfidence issue, where incorrect predictions are made with high confidence scores, hindering the applications in critical systems. In this paper, we propose a novel approach called Typicalness-Aware Learning (TAL) to address this issue and improve failure detection performance. 
We observe that, with the cross-entropy loss, model predictions are optimized to align with the corresponding labels via increasing logit magnitude or refining logit direction. However, regarding atypical samples, the image content and their labels may exhibit disparities. This discrepancy can lead to overfitting on atypical samples, ultimately resulting in the overconfidence issue that we aim to address. To tackle the problem, we have devised a metric that quantifies the typicalness of each sample, enabling the dynamic adjustment of the logit magnitude during the training process. By allowing atypical samples to be adequately fitted while preserving reliable logit direction, the problem of overconfidence can be mitigated. TAL has been extensively evaluated on benchmark datasets, and the results demonstrate its superiority over existing failure detection methods. Specifically, TAL achieves a more than 5% improvement on CIFAR100 in terms of the Area Under the Risk-Coverage Curve (AURC) compared to the state-of-the-art. Code is available at https://github.com/liuyijungoon/TAL.


### Installation
Execute the following commands sequentially to set up the environment:
```
conda create -n tal python=3.8
conda activate tal
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Training
The training process utilizes Weights & Biases ([wandb](wandb.ai)) for monitoring and visualization. Below is the training procedure using ImageNet dataset as an example.

#### Step 1: Configure 'sweep_mgpu.sh'
```
export WANDB_API_KEY=your_wandb_key
wandb sweep run_mgpu_imagenet.yaml
```

#### Step 2: Initialize wandb
```
sh sweep_mgpu.sh
```

You will receive output similar to:
```
wandb: Creating sweep from: run_mgpu_imagenet.yaml
wandb: Creating sweep with ID: bsvxxxxx
wandb: View sweep at: https://wandb.ai/your_account/your_project/sweeps/bsvxxxxx
wandb: Run sweep agent with: wandb agent your_account/your_project/bsvxxxxx
```

Then execute:
```
wandb agent your_account/your_project/bsvxxxxx
```

### Testing
Execute the following command to perform model testing:
```
python test.py --weights_path ./output/imagenet_resnet50_TAL_steplr_correct_first3_100_10.0_run1/epoch89/model.pth --fast 1
```
The 'cos' field in the test results represents the final confidence score.
