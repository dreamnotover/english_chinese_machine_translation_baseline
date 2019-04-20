# AI_Challenger  机器翻译
官方提供的脚本有不少错误，python脚本从2迁移到了3。 训练基本中去掉了batch_size项，改用 --worker_gpu_memory_fraction 可以免去内存溢出风险。

git clone https://github.com/dreamnotover/english_chinese_machine_translation_baseline.git
Neural Machine Translation (English-to-Chinese) baseline for AI_Challenger dataset.

# Requirenments

- python 3.6
- TensorFlow 1.12.0
- tensor2tensor
- jieba 0.39
 mkdir    t2t_tmp   t2t_data  raw_data
1、下载数据  链接: https://pan.baidu.com/s/18nRxRrUY0bOlYlPkCN2kyA 提取码: at2r 
解压后放入raw_data，所有官方数据都放入一个文件夹以方便处理
unzip  raw_data.zip   -C raw_data

2 定义新问题
 参考  https://blog.csdn.net/hpulfc/article/details/81172498
 新问题在./ai_data目录
 
3、语料预处理与向量化
sh  ./prepare.sh
sh  ./data_gen.sh

4、 训练模型 big模式在4台v100机器上训练，效果比base好多了
big 模式   sh  train_big.sh
base 模式   sh  train_base.sh

5、翻译
将t2t_data里面的字典文件拷贝到 ./dict_path （一定要用自己的）,然后预测。（因为翻译时只用到字典，也可指定t2t_data目录）
big 模式   sh  decode_big.sh
base 模式   sh  decode_base.sh

6、 提交结果 wrap_xml.pl后面跟三个参数，'<' 代表输入流， '>"  代表输出流 
 
./tools/wrap_xml.pl zh  t2t_tmp/ai_challenger_MTEnglishtoChinese_testA_20180827_en.sgm Wenhua < translation_testB_no_space.txt >submit.sgm

7、评估
sh    evaluate.sh
   
result文件夹放入了本人训练模型得出的结果。有点遗憾，最佳模型未能保留下来，t2t训练过程中会删除旧模型，有好的结果应该及时终止（或用另外一窗口拷贝）并保存好模型。


# References

Attention Is All You Need

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

Full text available at: https://arxiv.org/abs/1706.03762

Code availabel at: https://github.com/tensorflow/tensor2tensor

parameter   https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/flags.py
