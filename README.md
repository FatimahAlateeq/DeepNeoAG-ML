# DeeNeoAG: Neoantigen Epitope Prediction from Melanoma Antigens Using a Synergistic Deep Learning Model Combining Protein Language Models and Multi-Window Scanning Convolutional Neural Networks
Cheng-Che Chuang, Yu-Chen Liu, Yu-Yen Ou

## Abstract <a name="abstract"></a>
Neoantigens, derived from tumor-specific mutations, play a crucial role in eliciting anti-tumor immune responses and have emerged as promising targets for personalized cancer immunotherapy. Accurately identifying neoantigens from a vast pool of potential candidates is crucial for developing effective therapeutic strategies. This study presents a novel deep learning model that leverages the power of protein language models (PLMs) and multi-window scanning convolutional neural networks (CNNs) to predict neoantigens from normal tumor antigens with high accuracy. In this study, we present DeepName, a novel framwork combines the global sequence-level information captured by a pre-trained PLM with the local structural features extracted by a multi-window scanning CNN, enabling a comprehensive representation of the protein's mutational landscape. We demonstrate the superior performance of DeepName compared to existing methods and highlight its potential to accelerate the development of personalized cancer immunotherapies.
<br>

![workflow](https://github.com/B1607/DeepNeoAG/blob/f6085198d2cf4fffdca6564889b569e410b537ae/figure/figure_neo.png)

## Dataset <a name="Dataset"></a>

| Dataset                                        | Epitope Sequence          | Remove Similarity < 30% | Length <35            | Training Data (80%)       | Testing Data (20%)      |
|------------------------------------------------|---------------------------|-------------------------|-----------------------|---------------------------|-------------------------|
| Neoantigen                                     | 671                       | 303                     | 302                   | 241                       | 61                      | 
| Other                                          | 32206                     | 6717                    | 6710                  | 5367                      | 1343                    |
| Total                                          | 32877                     | 7019                    | 7012                  | 5608                      | 1404                    |

| Testing Data                                   | MHC I                     | MHC II                  |
|------------------------------------------------|---------------------------|-------------------------|
| Neoantigen                                     | 18                        | 43                      |
| Others                                         | 1005                      | 338                     |
| Total                                          | 1023                      | 281                     |

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
python get_dataset.py -in ../Data/Train -out ./Train -dt .prottrans -maxseq 35 #prottrans
python get_dataset.py -in ../Data/Train -out ./Train -dt .esm -maxseq 35 #esm
python get_dataset.py -in ../Data/Train -out ./Train -dt .npy -maxseq 35 #tape
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
