#! /bin/sh

#Download the dataset and put the dataset in ../raw_data file

DATA_DIR=./raw_data
TMP_DIR=./t2t_tmp
mkdir   $TMP_DIR  $DATA_DIR
  
#unwrap xml for valid data and test data
python tools/unwrap_xml.py $DATA_DIR/ai_challenger_MTEnglishtoChinese_validationset_20180823_zh.sgm >$TMP_DIR/valid.zh
python tools/unwrap_xml.py $DATA_DIR/ai_challenger_MTEnglishtoChinese_validationset_20180823_en.sgm >$TMP_DIR/valid.en

#Prepare Data

##Chinese words segmentation
python tools/jieba_cws.py $TMP_DIR/train.zh > $TMP_DIR/train-tok.zh
python tools/jieba_cws.py $TMP_DIR/valid.zh > $TMP_DIR/valid-tok.zh
#Tokenize and Lowercase English training data
chmod  a+x   tools/*.perl
cat $TMP_DIR/train.en | tools/tokenizer.perl -l en | tr A-Z a-z > $TMP_DIR/train-tok.en
cat $TMP_DIR/valid.en | tools/tokenizer.perl -l en | tr A-Z a-z > $TMP_DIR/valid-tok.en

#预测数据预处理
python tools/unwrap_xml.py  $DATA_DIR/ai_challenger_MTEnglishtoChinese_testA_20180827_en.sgm > $TMP_DIR/testA.en-zh.en 
#去掉头两列序号，只留下待翻译的句子
awk -F '\t' '{print $3}' $TMP_DIR/testA.en-zh.en > $TMP_DIR/to_pred_a.sgm
#所有大写转换成小写
cat $TMP_DIR/to_pred_a.sgm | tools/tokenizer.perl -l en | tr A-Z a-z > $TMP_DIR/to_pred_testA.sgm

python tools/unwrap_xml.py  $DATA_DIR/ai_challenger_MTEnglishtoChinese_testB_20180827_en.sgm > $TMP_DIR/testB.en-zh.en
awk -F '\t' '{print $3}' $TMP_DIR/testB.en-zh.en > $TMP_DIR/to_pred_b.sgm
#所有大写转换成小写
cat $TMP_DIR/to_pred_b.sgm | tools/tokenizer.perl -l en | tr A-Z a-z > $TMP_DIR/to_pred_testB.sgm
