## Masked Image/Text Pretraining

This code demonstrates how to train the MUSK model using unimodal image and text data, leveraging the unified mask modeling loss.

1. **Configure the Settings of `./configs/pretrain_musk_large.yaml`**
    - Set the paths for the preprocessed image and text files by assigning them to `--image_dir` and `--text_dir`. We provide example data [here](https://drive.google.com/drive/folders/1gaBMTnF4zVxt1hUn9qaZVsbXJeDp_-TH?usp=sharing).
    - Download the required text tokenizer `--tokenizer` [link](https://drive.google.com/file/d/1NJGch0cIhYzSSqTCJCRaCgJqDIG12d8H/view?usp=sharing) and specify its path.
    - Download the required image tokenizer `--tokenizer_weight` [link](https://drive.google.com/file/d/1fVxFnIPVZirEdg9tQ2vfv7MfEBOX9FuE/view?usp=sharing) and specify its path.
  
2. **Run the Pretraining Script**
    - cd `./scripts` and execute the pretraining script `bash ./run_pathology_pretrain.sh` to train the MUSK model. 
    - The `--HOST_NUM` parameter specifies the total number of machines; `--HOST_GPU_NUM` indicates the total number of GPUs per machine; `--CHIEF_IP` is the IP address of the master machine; `--INDEX` is the index of the machine; `--PORT_ID` is the port id of communication.

Monitor the log for `accuracy_mim` and `accuracy_mlm`. These metrics should increase steadily, indicating that the model is learning to recover the masked image tokens and text tokens.
