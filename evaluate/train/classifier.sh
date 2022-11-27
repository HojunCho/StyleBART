if [ -z "$1" ]
then
    echo "No dataset given"
    exit
fi

DATA=$1

if [ ! -d "data/preprocessed/$DATA" ]
then
    echo "There is no preprocessed data for $DATA; please run \"python -m style_bart.data.preprocess --$DATA\""
    exit
fi

if [ ! -d "content/eval/roberta" ]
then
    mkdir -p content/eval/roberta
    wget -O content/eval/roberta/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    wget -O content/eval/roberta/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    wget -O content/eval/roberta/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
    wget -O content/eval/roberta/roberta.large.tar.gz https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz 
    tar -zxvf content/eval/roberta/roberta.large.tar.gz --directory content/eval/roberta
    rm content/eval/roberta/roberta.large.tar.gz
fi

[ -f /.dockerenv ] && ENV=fairseq_docker || ENV=fairseq

if [ ! -d ".venv/$ENV" ]
then
    python -m venv ".venv/$ENV"
fi

source ".venv/$ENV/bin/activate"

pip show fairseq > /dev/null 2>&1 || pip install -r evaluate/requirements.txt

mkdir -p content/eval/$DATA/style/data-bin

for SPLIT in train dev; do
    if [ ! -f "content/eval/$DATA/style/data-bin/$SPLIT.input0" ]; then
        cat data/preprocessed/$DATA/sentences.$SPLIT.*.txt > content/eval/$DATA/style/data-bin/$SPLIT.input0
        > content/eval/$DATA/style/data-bin/$SPLIT.label
        for LABEL in 0 1; do
            yes $LABEL | head -n $(wc -l < data/preprocessed/$DATA/sentences.$SPLIT.$LABEL.txt) >> content/eval/$DATA/style/data-bin/$SPLIT.label
        done
    fi

    if [ ! -f "content/eval/$DATA/style/data-bin/$SPLIT.input0.bpe" ]; then
        python -m evaluate.train.multiprocessing_bpe_encoder \
            --encoder-json content/eval/roberta/encoder.json \
            --vocab-bpe content/eval/roberta/vocab.bpe \
            --inputs content/eval/$DATA/style/data-bin/$SPLIT.input0 \
            --outputs content/eval/$DATA/style/data-bin/$SPLIT.input0.bpe \
            --workers 60 \
            --keep-empty
    fi
done

if [ ! -d "content/eval/$DATA/style/data-bin/input0" ]; then
fairseq-preprocess \
    --only-source \
    --trainpref content/eval/$DATA/style/data-bin/train.input0.bpe \
    --validpref content/eval/$DATA/style/data-bin/dev.input0.bpe \
    --destdir content/eval/$DATA/style/data-bin/input0 \
    --workers 60 \
    --srcdict content/eval/roberta/dict.txt
fi

if [ ! -d "content/eval/$DATA/style/data-bin/label" ]; then
fairseq-preprocess \
    --only-source \
    --trainpref content/eval/$DATA/style/data-bin/train.label \
    --validpref content/eval/$DATA/style/data-bin/dev.label \
    --destdir content/eval/$DATA/style/data-bin/label \
    --workers 60
fi

MAX_SENTENCES=32         # Batch size.
UPDATE_FREQ=1
TOTAL_NUM_UPDATES=$(($(wc -l < content/eval/$DATA/style/data-bin/train.input0.bpe) * 10 / ($MAX_SENTENCES * $UPDATE_FREQ)))  # 10 epochs through Dataset for bsz
WARMUP_UPDATES=$((TOTAL_NUM_UPDATES * 6 / 100))      # 6 percent of the number of updates
echo $TOTAL_NUM_UPDATES $WARMUP_UPDATES
LR=1e-05                # Peak LR for polynomial LR scheduler.
HEAD_NAME=sentence_classification_head     # Custom name for the classification head.
NUM_CLASSES=2           # Number of classes for the classification task.
ROBERTA_PATH=content/eval/roberta/roberta.large/model.pt

fairseq-train content/eval/$DATA/style/data-bin \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --patience 0 \
    --update-freq $UPDATE_FREQ \
    --save-dir content/eval/$DATA/style
