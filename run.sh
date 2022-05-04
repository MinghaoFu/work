MODEL=RCAN
SCALE=4
SAVE=$MODEL
mkdir -p ./../experiment/${SAVE} ./../experiment/${SAVE}/history
LOG=./../experiment/${SAVE}/history/${MODEL}_BIX${SCALE}_`date +%Y.%m.%d_%H:%M:%S`.txt
touch ./../experiment/${SAVE}/README.md
CUDA_VISIBLE_DEVICES=3 python main.py  --resume -2 --model $MODEL --save $SAVE --scale $SCALE --decay_type step --lr 1e-6 --ext bin --chop 2>&1 | tee $LOG

