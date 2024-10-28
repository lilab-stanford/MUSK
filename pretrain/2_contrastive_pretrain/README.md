## Contrastive Image-Text Pretraining

This code demonstrates how to train the MUSK model using multimodal image-text pairs with contrastive loss.

1. **Configure the Settings in `./scripts/train.sh`**
    - Set the path of the masked pretrained model using the `--finetune` option.
    - Download the required text tokenizer and specify its path using `--sentencepiece_model`.
    - Download the Quilt1M dataset and set its path using `--DATA_Path`.
    - The `--HOST_NUM` parameter specifies the total number of machines; `--HOST_GPU_NUM` indicates the total number of GPUs per machine; `--CHIEF_IP` is the IP address of the master machine; `--INDEX` is the index of the machine; `--PORT_ID` is the port id of communication.
  
2. **Run the Pretraining Script**
    - Execute the script `./scripts/train.sh` to train the MUSK model.
    