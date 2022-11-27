# StyleBART
Source codes for `Rethinking Style Transformer with Energy-based Interpretation: Adversarial Unsupervised Style Transfer using a Pretrained Model`, accepted at EMNLP-22

## Environment Setting
### Using conda / pip
* Check `docker/requirements.txt` to install dependencies

### Using docker
* Check `Dockerfile` and `docker-compose.yml` to set up the environment.
* Append the below code in front of the python command instead of `CUDA_VISIBLE_DEVICES=<device_id>`:
```sh
# Non-docker version
# CUDA_VISIBLE_DEVICES=<device_id> python -m ...
CUDA_VISIBLE_DEVICES=0 python -m style_bart.train data=gyafc_fr

# Docker version
# USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose -f docker/docker-compose.yml run -e NVIDIA_VISIBLE_DEVICES=<device_id> app python -m ...
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose -f docker/docker-compose.yml run -e NVIDIA_VISIBLE_DEVICES=0 app python -m style_bart.train data=gyafc_fr
```

-----------------
## Folder description
* `.venv`: python environment. **This folder will be generated automatically.**
* `config`: configs for experiments
* `content`: forder for experiment outputs. **This folder will be generated automatically.**
    - `content/pretrain`: forder for pretraining
    - `content/main`: forder for main training
    - `content/eval`: forder for evaluation
* `data`: folder for train/dev/test data
    - `data/preprocess`: folder for preprocessed data. **This folder will be generated automatically.**
    - `data/yelp`: yelp dataset. `yelp_academic_dataset_review.json` should be included. Download from https://www.yelp.com/dataset
    - `data/gyafc`: gyafc dataset including `Entertainment_Music` and `Family_Relationships` folders. Download from https://github.com/raosudha89/GYAFC-corpus 
    - `data/amazon`: amazon dataset. Download from https://github.com/Nrgeup/controllable-text-attribute-transfer/tree/master/data/amazon  
* `docker`: docker configs
* `evaluate`: evaluation source code
* `style_bart`: StylaBART source code

-----------------
## Preprocessing
```sh
# python -m style_bart.data.preprocess [--dataset_name]
python -m style_bart.data.preprocess --gyafc --yelp --amazon
```

## Evalaution
Please check `evaluate/README.md`.
This procedure is also required to run below training code.

## Pretraining
### Classifier
```sh
# CUDA_VISIBLE_DEVICES=<device_id> python -m style_bart.pretrain.classifier data=<dataset_name> [args]
CUDA_VISIBLE_DEVICES=0 python -m style_bart.pretrain.classifier data=gyafc_fr
```

Depending on the dataset (especially for Amazon), classifier pretraining may not be converged.
In this case, larger batch size helps convergence.

```sh
CUDA_VISIBLE_DEVICES=0 python -m style_bart.pretrain.classifier data=amazon train.batch_size=512 # train.accumulation=2
```

### Autoencoder
```sh
# CUDA_VISIBLE_DEVICES=<device_id> python -m style_bart.pretrain.autoencoder data=<dataset_name> [args]
CUDA_VISIBLE_DEVICES=0 python -m style_bart.pretrain.autoencoder data=gyafc_fr
```

### Language models
```sh
# CUDA_VISIBLE_DEVICES=<device_id> python -m style_bart.pretrain.lm data=<dataset_name> label=<style> [args]
CUDA_VISIBLE_DEVICES=0 python -m style_bart.pretrain.lm data=gyafc_fr label=0
```
Language models should be trained for both labels 0 and 1.

## StyleBART Training 
```sh
# CUDA_VISIBLE_DEVICES=<device_id> python -m style_bart.train data=<dataset_name> [args]
CUDA_VISIBLE_DEVICES=0 python -m style_bart.train data=gyafc_fr # train.accumulation=2
```

-----------------
## StyleBART Inferencing 
### Downloading the trained model for each corpus
You can download the trained StyleBART weights from http://gofile.me/6XWMw/L53iBR52U

### Transfering the prompt or entire corpus
```sh
# CUDA_VISIBLE_DEVICES=<device_id> python -m style_bart.transfer -m <model_path> -l <target_style_label> <prompt>
CUDA_VISIBLE_DEVICES=0 python -m style_bart.transfer -m content/main/gyafc_fr/dump -l 0 "He loves you, too, girl...Time will tell."
```
Another option is redirecting the entire corpus to standard input
```sh
CUDA_VISIBLE_DEVICES=0 python -m style_bart.transfer -m content/main/gyafc_fr/dump -l 0 < data/preprocessed/gyafc_fr/sentences.test.1.txt
```

If you are using Docker, you need to add the `-T` option to redirect the corpus file.
```sh
docker compose -f docker/docker-compose.yml run -e NVIDIA_VISIBLE_DEVICES=0 -T app python -m style_bart.transfer -m content/main/gyafc_fr/dump -l 1 < data/preprocessed/gyafc_fr/sentences.test.0.txt > output.txt
```