#! /bin/sh

[ ! -s work-demo1 ] && mkdir work-demo1 
#验证数据预处理
python tools/unwrap_xml.py  demo/data1/src.sgm > work-demo1/src.en 
 #所有大写转换成小写
cat work-demo1/src.en | tools/tokenizer.perl -l en | tr A-Z a-z > work-demo1/to_pred_src.sgm

#翻译
DATA_DIR=./dict_path
TMP_DIR=./t2t_tmp
 
t2t-decoder --data_dir=${DATA_DIR} --problem=translate_enzh_sub92k --model=transformer \
--hparams_set=transformer_big_single_gpu --output_dir=./bigmodule \
-t2t_usr_dir=./ai_data --decode_hparams="beam_size=10,alpha=0.7" \
--decode_from_file=./work-demo1/to_pred_src.sgm --decode_to_file=work-demo1/hyp   \
--tmp_dir=${TMP_DIR}
#wrap your translation result into sgm file
 
./tools/wrap_xml.pl zh data1/src.sgm DemoSystem < work-demo1/hyp > work-demo1/hyp.sgm

#对结果分词

./tools/chi_char_segment.pl -t xml < work-demo1/hyp.sgm > work-demo1/hyp.seg.sgm 
./tools/chi_char_segment.pl -t xml < ./demo/data1/ref.sgm > work-demo1/ref.seg.sgm
 

#计算 BLEU score  result
 
./tools/mteval-v11b.pl -s ./demodata1/src.sgm -r work-demo1/ref.seg.sgm -t work-demo1/hyp.seg.sgm -c > work-demo1/bleu.log

#本次只计算了demo1的bleu，评估模型时可以使用三个demo数据计算结果的平均值作为评价标准
 

