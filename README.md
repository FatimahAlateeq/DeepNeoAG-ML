# MCNN_MC: Computational Prediction of Mitochondrial Carriers and Investigation of Bongkrekic Acid Toxicity using Protein Language Models and Convolutional Neural Networks
Muhammad Shahid Malik, Yan-Yun Chang, Yu-Chen Liu, Van The Le, Yu-Yen Ou

## Abstract <a name="abstract"></a>
Mitochondrial carriers (MCs) are essential proteins that transport metabolites across mitochondrial membranes, and play a critical role in cellular metabolism. ADP/ATP (Adenosine Diphosphate/Adenosine Triphosphate) is one of the most important carriers because it contributes to cellular energy production and is susceptible to the powerful toxin bongkrekic acid. A recent foodborne outbreak in Taipei, Taiwan, which caused four deaths and sickened 30 people, was caused by this toxin. The issue of bongkrekic acid poisoning has been a long-standing problem in Indonesia, with reports as early as 1895 detailing numerous deaths from contaminated coconut fermented cakes. Currently, there is no established computational method for identifying these carriers. Using a computational bioinformatics approach, we propose a novel method for predicting MCs from a broader class of secondary active transporters, with a focus on the ADP/ATP carrier and its interaction with bongkrekic acid. The proposed model combines Protein Language Models (PLMs) with multi-window scanning Convolutional Neural Networks (mCNNs). While PLM embeddings capture contextual information within proteins, mCNN scans multiple windows to identify potential binding sites and extract local features. The model achieved impressive results, with 96.66% sensitivity, 95.76% specificity, 96.12% accuracy, 91.83% Matthews Correlation Coefficient (MCC), 94.63% F1-Score, and 98.55% area under the curve (AUC). The results demonstrate the effectiveness of the proposed approach in predicting MCs and elucidating their functions, particularly in the context of bongkrekic acid toxicity. This study provides a valuable tool for identifying novel mitochondrial complexes, characterizing their functional roles, and understanding mitochondrial toxicology mechanisms. By utilizing computational methods to improve our understanding of cellular processes and drug-target interactions, these findings are contributing to the development of therapeutic strategies for mitochondrial disorders, reducing the devastating effects of bongkrekic acid poisoning.

<br>

![workflow](https://github.com/B1607/DeepNeoAG/blob/0c30a2ba1b9357d52766d6402e68b507441c4fa1/figure/flowchart.png)

## Dataset <a name="Dataset"></a>

| Dataset                             | Original Protein Sequence | Remove Similarity < 30% | Training Data            | Validation Data       |
|-------------------------------------|---------------------------|-------------------------|--------------------------| --------------------- |
| Secondary Activate Transporter      | 290                       | 256                     | 202                      | 51                    |
| Mitochondrial Carrier               | 282                       | 136                     | 108                      | 28                    |
| Total                               | 527                       | 389                     | 310                      | 79                    |


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
python get_dataset.py -in ../Data/Train -out ./Train -dt .prottrans -maxseq 800
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
