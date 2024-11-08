# Financial Question Answering Web Application with Fine-Tuned LLaMA
Tecnical task set by Tomoro to do something interesting with the ConvFinQA dataset

## Setup environment
Setup the environment for training and the web app
 - `python -m venv .venv`
 - `source .venv/bin/activate`
 - `pip install -r requirements.txt`

You will also need to login to huggingfaces following instructions here (https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) 

## Setup for Fine-Tuning
Follow these instructions to get set up to fine tune a model:

1. Instantiate to raw dataset
    - `git submodule init ConvFinQA/`
    - `git pull --recurse-submodules`

2. Preprocess data for finetuning
    - `./setup_data.sh`

3. Run Training
    - `python train_model.py`

## Run web app
Follow these instructions to run the web app

1. (Optional) Unzip prepackaged model I trained
    - `unzip llama-finetuned-convfinqa.zip`

2. Run Flask App (for local testing)
    - `flask --app server --debug`

3. Run Flask App (for deployment)
    - `flask --app server --host=0.0.0.0 --port=80`
