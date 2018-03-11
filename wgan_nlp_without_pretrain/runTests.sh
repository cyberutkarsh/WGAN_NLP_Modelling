count=1
while [ $count -le 22 ];do
python generate.py --CKPT_PATH=/data/projects/rnn.wgan2/logs/Generator_GRU_CL_VL_TH-Discriminator_GRU-50-10-512-512-1519669065.6-/checkpoint/seq-"$count"/ckp
python evaluate.py --INPUT_SAMPLE=output/sample.txt
(( count++ ))
done
