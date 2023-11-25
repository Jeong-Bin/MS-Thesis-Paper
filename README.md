### Robust Few-Shot Classification via Bilevel Knowledge Distillation ([Korean Paper](https://github.com/Jeong-Bin/MS-Thesis-Paper/files/13464189/paper.pdf))

Mini-ImageNet-C, CUB-200-C, FGVC-Aircraft-C를 위한 corruption 이미지는  ```./data/custom_create_FSL_C.py```로 생성할 수 있고, CIFAR-FS-C를 위한 corruption 이미지는 ```./data/custom_create_cifar_C.py```로 생성할 수 있습니다. <br/>
이 [Github](https://github.com/hendrycks/robustness/tree/master)의 ```ImageNet-C/create_c```에 위 코드들을 추가하여 사용하는 것을 추천합니다. <br/>
그리고 원본 데이터셋이 준비되어 있어야 합니다. <br/>

You can create **corrupt images** with ```./data/custom_create_FSL_C.py``` for Mini-ImageNet-C, CUB-200-C, FGVC-Aircraft-C or ```./data/custom_create_cifar_C.py``` for CIFAR-FS-C. <br/>
I recommend adding the above codes to the directory in ```ImageNet-C/create_c``` of this [Github](https://github.com/hendrycks/robustness/tree/master). <br/>
And original datasets should be prepared.

<br/>

**[Download link of original datasets]**
- Mini-ImageNet : [Github](https://github.com/yaoyao-liu/mini-imagenet-tools) or [Kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet).
- CUB-200-2011 : [Github](https://github.com/pytorch/vision/issues/2992) or [Caltech](https://data.caltech.edu/records/65de6-vp158).
- CIFAR-FS : [Github](https://github.com/bertinetto/r2d2) or [Drive](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view).
- FGVC-Aircraft : [Site](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
