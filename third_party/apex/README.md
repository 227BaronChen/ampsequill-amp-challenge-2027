# APEX pathogen - predicting species-specific antimicrobial activity against pathogens using deep learning

## Predict AMPs using APEX
By running 
`APEX_predict.py -i ./test_seqs.fasta -g 1`, species-specific antmicrobial activties (MICs) of peptides in ./test_seqs.fasta will be generated and saved in Predicted_MICs.csv.

You can replace `./test_seqs.fasta` with your own peptide FASTA file for making new predictions. The option `-g 1` enables GPU mode for APEX. To run the model in CPU mode, simply use `-g 0`. 

Pathogen list: 

A. baumannii ATCC 19606	 

E. coli ATCC 11775	

E. coli AIC221	

E. coli AIC222	

K. pneumoniae ATCC 13883	

P. aeruginosa PA01	

P. aeruginosa PA14	

S. aureus ATCC 12600	

S. aureus (ATCC BAA-1556) - MRSA	

vancomycin-resistant E. faecalis ATCC 700802	

vancomycin-resistant E. faecium ATCC 700221

You can also upload peptide fasta file to https://apexdb.ai/ for MIC prediction.

## File description
Folder APEX_pathogen_models contains the pretrained APEX models (8 in total)

APEX_models.py: defines the neural network architecture of APEX

utils.py: helper functions

aaindex1.csv: amino acid embeddings (freezed during training, obtained from https://www.genome.jp/aaindex/)

test_seqs.fasta: example FASTA input of APEX_predict.py (peptides should be <= 50 amino acids)

APEX_predict.py: predict species-specific antmicrobial activties (minimum inhibitory concentration [MIC]; unit: uM) against 11 pathogens for peptides in test_seqs.txt. We use 8 APEX models trained under different neural network architectures or training strategies to make predictions. Predictions from the base learners will be averaged to generate the final activity prediction. 



## Software version
pytorch: 1.11.0+cu113

## Set up environment
`conda create -n apex_env python==3.9`

`conda activate apex_env`

`pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113`

`pip install numpy==1.23.2`

`pip install pandas==2.0.2`

`pip install scikit-learn==1.4.2`

`pip install scipy==1.10`

`pip install biopython==1.82`

## Contacts
If you have any questions or comments, please feel free to email Fangping Wan (fangping[dot]wan[at]pennmedicine[dot]upenn[dot]edu) and/or César de la Fuente (cfuente[at]pennmedicine[dot]upenn[dot]edu).

