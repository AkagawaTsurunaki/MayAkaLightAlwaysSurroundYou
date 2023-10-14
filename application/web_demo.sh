PRE_SEQ_LEN=128

CUDA_VISIBLE_DEVICES=0 python web_demo.py \
    --model_name_or_path 'D:\AkagawaTsurunaki\Models\chatglm-6b-int8' \
    --ptuning_checkpoint 'D:\AkagawaTsurunaki\WorkSpace\PycharmProjects\MayAkaLightAlwaysSurroundYou\models\akako\Akako-int8-4.0Msamples\checkpoint-17850' \
    --pre_seq_len $PRE_SEQ_LEN \

