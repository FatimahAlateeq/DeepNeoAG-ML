# DeepNAME: Neoantigen Epitope Prediction from Melanoma Antigens Using a Synergistic Deep Learning Model Combining Protein Language Models and Multi-Window Scanning Convolutional Neural Networks

## Abstract <a name="abstract"></a>
Neoantigens, derived from tumor-specific mutations, play a crucial role in eliciting anti-tumor immune responses and have emerged as promising targets for personalized cancer immunotherapy. Accurately identifying neoantigens from a vast pool of potential candidates is crucial for developing effective therapeutic strategies. This study presents a novel deep learning model that leverages the power of protein language models (PLMs) and multi-window scanning convolutional neural networks (CNNs) to predict neoantigens from normal tumor antigens with high accuracy. In this study, we present DeepName, a novel framwork combines the global sequence-level information captured by a pre-trained PLM with the local structural features extracted by a multi-window scanning CNN, enabling a comprehensive representation of the protein's mutational landscape. We demonstrate the superior performance of DeepName compared to existing methods and highlight its potential to accelerate the development of personalized cancer immunotherapies.
<br>

![workflow](https://github.com/B1607/DeepNeoAG/blob/0c30a2ba1b9357d52766d6402e68b507441c4fa1/figure/flowchart.png)

## Dataset <a name="Dataset"></a>

| Dataset                             | Epitope Sequence          | Remove Similarity < 30% | Length <35            |
|-------------------------------------|---------------------------|-------------------------|-----------------------| 
| Neoantigen                          | 671                       | 303                     | 302                   | 
| Viral antigen                       | 32206                     | 6716                    | 6710                  | 
  Germline/ Self/ Host antigen                       
| Total                               | 32877                     | 7019                    | 7012                  |


## Quick start <a name="quickstart"></a>

### Step 1: Generate Data Features

Navigate to the data folder and utilize the FASTA file to produce additional data features, saving them in the dataset folder.

Example usage:
```bash
python get_ProtTrans.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_tape.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_esm.py "Pretrained model of ESM" "Your FASTA file folder" "The destination folder of your output" --repr_layers 33 --include per_tok
```
Alternative example:
```bash
python get_ProtTrans -in ./Fasta/Train -out ./Train
python get_tape -in ./Fasta/Train -out ./Train
python get_esm.py esm2_t33_650M_UR50D ./Fasta/Train ./Train --repr_layers 33 --include per_tok
```

### Step 2: Generate Dataset Using Data Features

Transition to the dataset folder and utilize the data features to produce a dataset.

Example usage:
```bash
python get_dataset.py -in "Your data feature Folder" -out "The destination folder of your output"  -dt "datatype" -maxseq "Setting of Sequence length."

```
Alternative example:
```bash
python get_dataset.py -in ../Data/Train -out ./Train -dt .prottrans -maxseq 800 #prottrans
python get_dataset.py -in ../Data/Train -out ./Train -dt .esm -maxseq 800 #esm
python get_dataset.py -in ../Data/Train -out ./Train -dt .npy -maxseq 800 #tape
```

### Step 3: Execute Prediction

Navigate to the code folder to execute the prediction.

Command-line usage:
```bash
python main.py 
```
Alternatively, utilize the Jupyter notebook:
```bash
main.ipynb
```
