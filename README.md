## TAL - Official PyTorch Implementation
![image](https://github.com/user-attachments/assets/e0f6294a-54f8-45b9-8a29-d78865df930a)


### [NeurIPS 2024] Typicalness-Aware Learning for Failure Detection
Yijun Liu, Jiequan Cui, Zhuotao Tian*, Senqiao Yang, Qingdong He, Xiaoling Wang, and Jingyong Su*<br>

[Paper](https://neurips.cc/virtual/2024/poster/95120)

### Abstract
Deep neural networks (DNNs) often suffer from the overconfidence issue, where incorrect predictions are made with high confidence scores, hindering the applications in critical systems. In this paper, we propose a novel approach called Typicalness-Aware Learning (TAL) to address this issue and improve failure detection performance. 
We observe that, with the cross-entropy loss, model predictions are optimized to align with the corresponding labels via increasing logit magnitude or refining logit direction. However, regarding atypical samples, the image content and their labels may exhibit disparities. This discrepancy can lead to overfitting on atypical samples, ultimately resulting in the overconfidence issue that we aim to address. To tackle the problem, we have devised a metric that quantifies the typicalness of each sample, enabling the dynamic adjustment of the logit magnitude during the training process. By allowing atypical samples to be adequately fitted while preserving reliable logit direction, the problem of overconfidence can be mitigated. TAL has been extensively evaluated on benchmark datasets, and the results demonstrate its superiority over existing failure detection methods. Specifically, TAL achieves a more than 5% improvement on CIFAR100 in terms of the Area Under the Risk-Coverage Curve (AURC) compared to the state-of-the-art. Code is available at https://github.com/liuyijungoon/TAL.

### Others
The refined code will be coming in a month.
