# DroidCoder: Enhanced Android Code Completion with Context-Enriched Retrieval-Augmented Generation

## Dependencies

python==3.9.18

accelerate==0.27.2

bitsandbytes==0.42.0

editdistance==0.8.1

fuzzywuzzy==0.18.0

numpy==1.26.4

peft==0.9.0

sentence-transformers==2.5.1

torch==2.0.1

transformers==4.39.0

## Fine-Tuning

You can download checkpoints for `CodeT5+ 220M/770M`, `CodeGPT` and `StarCoderBase-1B` from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.11419821) or you can run:

```sh
cd AndroidCompletion
python finetune.py

# optional arguments:
#  -h, --help              show this help message and exit
#  --checkpoint CHECKPOINT Model checkpoint
#  --batchsize BATCHSIZE   Batch size
#  --epochs EPOCHS         Epoch Nums
#  --resume                Resume or not
```

## Inference

You can directly view results that stored in folder `predictions`, or you can run:

```sh
cd AndroidCompletion
python generate.py --all

# optional arguments:
#  -h, --help              show this help message and exit
#  --checkpoint CHECKPOINT Model checkpoint
#  --new                 Generate Unknown APP Type Predictions
#  --codegpt             Generate CodeGPT Model Predictions
#  --peft                Generate StarCoderBase Model Predictions
```

## Evaluation

All results data are in folder `AndroidCompletion/visualization`.

For EM and ES, you can run:

```sh
cd AndroidCompletion
python compute_score.py
```

And the *Compilation Assurance Rate* is calculated in folder `DataProcessAndroid` and specifically in file `PostProcess.java`.

## Data Processing

The folder `AndroidCompletion/datasets` contains all the data which was already preprocessed, and can be directly used to fine-tune, infer or evaluate.

The main workflow for data processing:

> Please note that the current code is complete but still being organized and a public and more elegant version will be updated after the review.

- The data of 50 application repositories is in folder `dataset`, please download the data from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.11419821) before the following process.

- Constructing retrieval database and generating code completion cases are in folder `DataProcessAndroid` and specifically in file `Builder.java`.

- Component Methods Classification is in folder `DataProcessAndroid` and specifically in file `Classifier.java`.

- Relevant Retrieval you can run:

  ```sh
  cd AndroidCompletion
  python prepare_dataset.py --sim
  ```

- Contextual Enhancement is in folder `DataProcessAndroid` and specifically in file `CallMain.java`.

- Final integration you can run:

  ```sh
  cd AndroidCompletion
  python prepare_dataset.py --integrate
  ```

  