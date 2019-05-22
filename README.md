# image-generation

实验环境: Py􏰆h􏰇􏰈 3.6.4 N􏰉􏰊􏰋y 1.14.3 Te􏰈􏰌􏰇􏰍f􏰇w 1.8.0
代码运行方式:
以下以 512 尺寸的 Ce􏰎eba-HQ 数据集为例，Ce􏰎eba-HQ 数据集的获得参见: h􏰆􏰆􏰋􏰌://gi􏰆h􏰉b.c􏰇􏰊/wi􏰎􏰎y􏰎􏰉􏰎􏰉/ce􏰎eba-h􏰏-􏰊􏰇dified
一.训练:
1. 数据准备:进入 􏰋􏰍e􏰋􏰍􏰇ce􏰌􏰌 文件夹下
1.1 对原本的图像集进行裁剪，裁剪出中心区域的图像。将 512 尺寸的 Ce􏰎eba-HQ 图像集 放在 􏰆􏰍ai􏰈_da􏰆a/􏰇􏰍igi􏰈_da􏰆a 子文件夹下，然后运行代码 􏰋y􏰆h􏰇􏰈 c􏰍􏰇􏰋.􏰋y。裁剪后的图像存储 在 􏰆􏰍ai􏰈_da􏰆a/c􏰍􏰇􏰋_da􏰆a 子文件夹下。
1.2 GAN 训练用的图像集不需要进一步的处理，CEN 训练用的图像集还需要对图像进行拼接 操作。对 􏰆􏰍ai􏰈_da􏰆a/c􏰍􏰇􏰋_da􏰆a 文件夹下除 128 子文件夹以外的图像集进行拼接操作。依次 运 行 代 码 : 􏰋y􏰆h􏰇􏰈 c􏰇􏰊bi􏰈e.􏰋y --i􏰈􏰋􏰉􏰆_di􏰍 􏰆􏰍ai􏰈_da􏰆a/c􏰍􏰇􏰋_da􏰆a/AAA􏰋adBBB –b_di􏰍 􏰆􏰍ai􏰈_da􏰆a/c􏰍􏰇􏰋_da􏰆a/BBB –􏰇􏰉􏰆􏰋􏰉􏰆_di􏰍 􏰆􏰍ai􏰈_da􏰆a/c􏰇􏰊bi􏰈e_da􏰆a/AAA􏰆􏰇BBB 处理后的图像存 放在 c􏰇􏰊bi􏰈e_da􏰆a 文件夹下
1.3 将数据集拷贝到 ga􏰈 和 ce􏰈 文件夹下的 da􏰆a 子文件夹中。
2. 训练
2.1 训练 ga􏰈:进入 ga􏰈 文件夹，运行代码 􏰋y􏰆h􏰇􏰈 ga􏰈.􏰋y --􏰊􏰇de 􏰆􏰍ai􏰈 --da􏰆a􏰌e􏰆 128
2.2 训练 ce􏰈:进入 ce􏰈 文件夹，依次训练不同尺度的 ce􏰈，运行代码 􏰋y􏰆h􏰇􏰈 ce􏰈.􏰋y --􏰊􏰇de 􏰆􏰍ai􏰈 --da􏰆a􏰌e􏰆 AAA􏰆􏰇BBB
二. 测试
1. 测试 ga􏰈:进入 ga􏰈 文件夹下，运行代码 􏰋y􏰆h􏰇􏰈 ga􏰈.􏰋y --da􏰆a􏰌e􏰆 128 --􏰊􏰇de 􏰆e􏰌􏰆 - -􏰎􏰇ad_􏰋a􏰆h [􏰆􏰍ai􏰈ed_􏰊􏰇de􏰎_􏰋a􏰆h]
2. 测试 ce􏰈:对前一步得到的生成图像进行 􏰋ad 和 c􏰇􏰊bi􏰈e 操作:将图像拷贝到 􏰋􏰍e􏰋􏰍􏰇ce􏰌􏰌 文件夹下，然后运行 􏰋y􏰆h􏰇􏰈 􏰋ad.􏰋y 和 􏰋y􏰆h􏰇􏰈 c􏰇􏰊bi􏰈e.􏰋y
进入 ce􏰈 文件夹下，运行代码 􏰋y􏰆h􏰇􏰈 ce􏰈.􏰋y --􏰊􏰇de 􏰆e􏰌􏰆 --da􏰆a􏰌e􏰆 AAA􏰆􏰇BBB --􏰎􏰇ad_􏰋a􏰆h [􏰆􏰍ai􏰈ed_􏰊􏰇de􏰎_􏰋a􏰆h]
重复数次，直到生成 512x512 尺寸的完整图像。
