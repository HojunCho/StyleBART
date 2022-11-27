echoerr() { echo "$@" 1>&2; }

while getopts d:s:t: opt; do
    case $opt in
        d) DATA=$OPTARG;;
        s) SPLIT=$OPTARG;;
        t) TEMP=$OPTARG;;
        *) echo "$opt is not the option" || exit;;
    esac
done

shift $[ $OPTIND - 1 ]

if [ -z "$DATA" ]; then
    echoerr 'Missing data option -d'
    exit 1
elif [ ! -d "data/preprocessed/$DATA" ]; then
    echoerr "No data: $DATA"
    exit 1
elif [ -z "$SPLIT" ]; then
    echoerr 'Missing split option -s'
    exit 1
elif [ -z "$TEMP" ]; then
    TEMP='./temp'
    mkdir -p $TEMP
fi

if (($# != 2)); then
    echoerr 'Please provide the transferred corpus 0->1, 1->0'
    exit
elif [ ! -f $1 ]; then
    echoerr 'Invalid path for' $1
    exit
elif [ ! -f $2 ]; then
    echoerr 'Invalid path for' $2
    exit
fi

LABEL_PATH=content/eval/$DATA/style/$SPLIT.label.txt
if [ ! -f $LABEL_PATH ]; then
    > $LABEL_PATH
    for LABEL in 0 1; do
        yes $((1 - LABEL)) | head -n $(wc -l < data/preprocessed/$DATA/sentences.$SPLIT.$LABEL.txt) >> $LABEL_PATH
    done
fi

mkdir -p $TEMP
TEMP=$TEMP/temp

cat $@ > $TEMP

[ -f /.dockerenv ] && ENV=fairseq_docker || ENV=fairseq
if [ ! -d ".venv/$ENV" ]; then
    python -m venv ".venv/$ENV"
fi
source ".venv/$ENV/bin/activate"
pip show fairseq > /dev/null 2>&1 || { pip install wheel 1>&2; pip install -r evaluate/requirements.txt 1>&2; }

echoerr "RoBERTa $DATA classification"
ACC=$(python -m evaluate.roberta_classify \
    --input_file $TEMP --label_file $LABEL_PATH \
    --model_dir content/eval/$DATA/style --model_data_dir content/eval/$DATA/style/data-bin)

echoerr "RoBERTa CoLA acceptability classification"
COLA=$(python -m evaluate.acceptability \
    --input_file $TEMP \
    --model_dir content/eval/fluency --model_data_dir content/eval/fluency/cola-bin)

if [ ! -f "data/preprocessed/$DATA/reference.$SPLIT.0.txt" ]; then
    for corpus in $(basename -a $(ls data/preprocessed/$DATA/reference.$SPLIT.0.?.txt) | cut -d '.' -f 4)
    do
        cat data/preprocessed/$DATA/reference.$SPLIT.0.$corpus.txt data/preprocessed/$DATA/reference.$SPLIT.1.$corpus.txt \
          > data/preprocessed/$DATA/reference.$SPLIT.$corpus.txt
    done
fi

echoerr "Paraphrase scores"
SIM=$(python -m evaluate.get_paraphrase_similarity \
    --generated_path $TEMP --reference_strs \
    $(basename -a $(ls data/preprocessed/$DATA/reference.$SPLIT.?.txt) | awk -F '.' '{ print "ref" $3 }' | tr '\n' ',' | sed 's/,$//') \
    --reference_paths $(ls data/preprocessed/$DATA/reference.$SPLIT.?.txt | tr '\n' ',' | sed 's/,$//') \
    --output_path $TEMP.generated_vs_gold.txt --store_scores)

deactivate

echoerr "Perplexity"
PPL0=$(python -m evaluate.perplexity data=$DATA label=1 target=$1)
PPL1=$(python -m evaluate.perplexity data=$DATA label=0 target=$2)
echoerr "0->1: $PPL0, 1->0: $PPL1"
PPL=$(echo "($PPL0 + $PPL1)/2" | bc -l)
echoerr "Avg perplexity = $PPL"

echoerr "Normalized paraphrase score"
ALL=($(python -m evaluate.micro_eval \
                --generated_file $TEMP \
                --classifier_file $TEMP.roberta_labels \
                --paraphrase_file $TEMP.pp_scores \
                --acceptability_file $TEMP.acceptability_labels))

echoerr "ACC, SIM, J(ACC,SIM), FL-CoLA, FL-PPL"
echo $ACC,$SIM,${ALL[0]},$COLA,$PPL
