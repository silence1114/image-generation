# image-generation

#### 实验环境：<br>
Python 3.6.4<br>
Numpy 1.14.3<br>
Tensorfow 1.8.0<br><br><br>

#### 代码运行方式：<br>
以下以512尺寸的Celeba-HQ数据集为例，Celeba-HQ数据集的获得参见： <br>
https://github.com/willylulu/celeba-hq-modified<br><br><br>

## 一．训练：<br>
### 1.	数据准备：进入preprocess文件夹下<br>
1.1 对原本的图像集进行裁剪，裁剪出中心区域的图像。将512尺寸的Celeba-HQ图像集放在train_data/origin_data子文件夹下，然后运行代码 python crop.py。裁剪后的图像存储在train_data/crop_data子文件夹下。<br>
1.2 GAN训练用的图像集不需要进一步的处理，CEN训练用的图像集还需要对图像进行拼接操作。对train_data/crop_data文件夹下除128子文件夹以外的图像集进行拼接操作。依次运行代码： python combine.py --input_dir train_data/crop_data/AAApadBBB –b_dir train_data/crop_data/BBB –output_dir train_data/combine_data/AAAtoBBB 处理后的图像存放在combine_data文件夹下<br>
1.3 将数据集拷贝到gan和cen文件夹下的data子文件夹中。<br><br>

### 2.	训练<br>
2.1 训练gan：进入gan文件夹，运行代码 python gan.py --mode train --dataset 128 <br>
2.2 训练cen：进入cen文件夹，依次训练不同尺度的cen，运行代码python cen.py --mode train --dataset AAAtoBBB<br><br><br>

## 二. 测试<br>
1. 测试gan：进入gan文件夹下，运行代码python gan.py  --dataset 128 --mode test  --load_path [trained_model_path]<br>
2. 测试cen：对前一步得到的生成图像进行pad和combine操作：将图像拷贝到preprocess文件夹下，然后运行python pad.py 和python combine.py <br>
进入cen文件夹下，运行代码python cen.py --mode test --dataset AAAtoBBB --load_path [trained_model_path]
重复数次，直到生成512x512尺寸的完整图像。<br>
