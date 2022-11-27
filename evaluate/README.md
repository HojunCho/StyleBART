# Style Transfer Evaluation
This README and the evaluation code are based from https://github.com/martiansideofthemoon/style-transfer-paraphrase/tree/master/style_paraphrase/evaluation

Only Python=3.8.* is competible for evaluation. Otherwise, please use Docker.

### Using Docker
* Check `Dockerfile` and `docker-compose.yml` to set up the environment.
* Append the below code in front of the python/bash command instead of `CUDA_VISIBLE_DEVICES=<device_id>`:
```sh
# Non-docker version
# CUDA_VISIBLE_DEVICES=<device_id> ...
CUDA_VISIBLE_DEVICES=0 bash evaluate/eval.sh

# Docker version
# USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose -f docker/docker-compose.yml run -e NVIDIA_VISIBLE_DEVICES=<device_id> app ...
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose -f docker/docker-compose.yml run -e NVIDIA_VISIBLE_DEVICES=0 app bash evaluate/eval.sh
```

## Evaluation Preparation
### Accuracy
We use RoBERTa-large classifiers to check style transfer accuracy.
The training code are given in `/evaluate/train/classifier.sh`.
```bash
# Run it on the project root directory
CUDA_VISIBLE_DEVICES=0 bash evaluate/train/classifier.sh $DATA # amazon yelp gyafc_fr gyafc_em 
```
Before training the classicier, the preprocessed data should be prepared in `/data/preprocessed`
The trained classifier and relevant data will be placed in `/content/eval/$DATA/accuracy`

## Train RoBERTa classifier for evaluation
```sh
# CUDA_VISIBLE_DEVICES=<device_id> python -m evaluate.train.classifier data=<dataset_name> [args]
CUDA_VISIBLE_DEVICES=0 python -m evaluate.train.classifier data=gyafc_fr
```

### Similarity
We use the SIM model from Wieting et al. 2019 ([paper](https://www.aclweb.org/anthology/P19-1427/)) for our evaluation.
The code for similarity can be found under `similarity`. Make sure to download the `evaluation/sim` folder from the [Google Drive link](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing) and place it as `/content/eval/similarity`.

### Fluency (CoLA Classifier)
We use a RoBERTa-large classifier trained on the [CoLA corpus](https://nyu-mll.github.io/CoLA) to evaluate fluency of generations.
We leverages the same classifier used by Kalpesh et al. 2020 ([paper](https://aclanthology.org/2020.emnlp-main.55/))
Make sure to download the `evaluation/cola_classifier` folder from the [Google Drive link](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing) and place it as `/content/eval/fluency`.

### Fluency (Language Model Perplexity)
In addition to CoLA classifier, we use a fine-tunned GPT-2 language model trained on each corpus to evaluate fluency.
The training code are given in `/evaluate/train/lm.py`.
```sh
# CUDA_VISIBLE_DEVICES=<device_id> python -m evaluate.train.lm data=<dataset_name> label=<style> [args]
CUDA_VISIBLE_DEVICES=0 python -m evaluate.train.lm data=gyafc_fr label=0
```

## Running Evaluation for specific outputs
You can evalute the transfered outputs by the below command.
```sh
# CUDA_VISIBLE_DEVICES=<device_id> bash evaluate/eval.sh -d <dataset_name> -s <split> <0->1 transfered text file> <1->0 transfered text file>
CUDA_VISIBLE_DEVICES=0 bash evaluate/eval.sh -d gyafc_fr -s test content/main/gyafc_fr/default/out/0to1.txt content/main/gyafc_fr/default/out/1to0.txt
```
