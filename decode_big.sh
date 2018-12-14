DATA_DIR=./dict_path
TMP_DIR=./t2t_tmp
#翻译
t2t-decoder --data_dir=${DATA_DIR} --problem=translate_enzh_sub92k --model=transformer \
--hparams_set=transformer_big_single_gpu --output_dir=./bigmodule \
-t2t_usr_dir=./ai_data --decode_hparams="beam_size=10,alpha=0.6" \
--decode_from_file=${TMP_DIR}/to_pred_testA.sgm --decode_to_file=translation_testA.txt   \
--tmp_dir=${TMP_DIR}
#去空格
sed -r 's/\s+//g' translation_testA.txt > translation_testA_no_space.txt

paste  -d"\t"  ${TMP_DIR}/to_pred_testA.sgm   translation_testA_no_space.txt >  Paralle_TestA_big.txt

DATA_DIR=./dict_path
TMP_DIR=./t2t_tmp
#翻译
t2t-decoder --data_dir=${DATA_DIR} --problem=translate_enzh_sub92k --model=transformer \
--hparams_set=transformer_big_single_gpu --output_dir=./bigmodule \
-t2t_usr_dir=./ai_data --decode_hparams="beam_size=10,alpha=0.7" \
--decode_from_file=${TMP_DIR}/to_pred_testB.sgm --decode_to_file=translation_testB.txt   \
--tmp_dir=${TMP_DIR}
#去空格
sed -r 's/\s+//g' translation_testB.txt > translation_testB_no_space.txt

paste  -d"\t"  ${TMP_DIR}/to_pred_testB.sgm   translation_testB_no_space.txt >  Paralle_TestB_big.txt

