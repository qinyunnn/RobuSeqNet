# SRNet



## Install


### Install for training 

- Clone the repo
``` sh
git clone https://github.com/qinyunnn/Multi-Read-Reconstruction.git
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n wenet python=3.8
conda activate SRNet
pip install -r requirements.txt
```



## Usage

We provided a training example in examples.

## Data Preparation
```
# put your dataset under data/ directory with the same structure shown in the example/data/

data
 |-reads.txt
 |-reference.txt
```

### Notices:
* In reads.txt, the adjacent two clusters are segmented using '==============================='. For example:
```
AACCAATACCTTGAACCTAACTCGAGTTAACAAACGCAATTCACAGAACAAGGACGTCGGACGGTGTCCAGAATACCGGCCTCGTGACCGTGGCCAGGGAACCTGACAATGTCAGGCCTTACCGACACACGCAACCTCTTGCTGAAAGGCCT
AACCAATACCTTGAACCTAACTCGAGTTAACAAACGCAATTCACAGAACAAGGACGTCGGACGGTGTCCAGAATACCGGCCTCGTGACCGTGGCCAGGGAACCTGACAATGTCAGGCCTTACCGACAACGCAACCTCTTGCTGAAAGGCCT
AACCAATACCTTGAACCTAACTCGAGTTAACAAACGCAATTCACAGAACAAGGACGTCGGACGGTGTCCAGAATACCGGCCTCGTGACCGTGGCCAGGGAACCTGACAATGTCAGGCCTTACCGACACACGCAACCTCTTGCTGAAAGGCCT
AACCAATACCTTGAACCTAACTCGAGTTAACAAACGCAATTCACAGAACAAGGACGTCGGACGGTGTCCAGAATACCGGCCTCGTGACCGTGGCCAGGGAACCTGACAATGTCAGGCCTTACCGACACACGCAACCTCTTGCTGAAAGGCCT
===============================
AATACCTTGAAGTCACTGGTACTGAACATGCCTTCTGACCGTTAGGACTTATCTTCCTGTCGGTATAAGATCTACTACTACAACACTGGTTTCAACTAGCGGGAGAAGTCCTTACCGAGTTCTGCGGCTGGCTGATAGCGTGTGCCCTCTGG
AATACCTTGAAGTCACTGGTACTGAACATGCCTTCTGACCGTTAGGACTTATCTTCCTGTCGGTATAAGATCTACTACTACAACACTGGTTTCAACTAGCGGGAGAAGTCCTTACCGAGTTCTGCGGCTGGCTGATAGCGTGTGCCCTCTGG
AATACCTTGAAGTCACTGGTACTGAACATGCCTTCTGACCGTTAGGACTTATCTTCCTGTCGGTATAAGATCTACTACTACAACACTGGTTTCAACTAGCGGGAGAAGTCCTTACCGAGTTCTGCGGCTGGCTGATAGCGTGTGCCCTCTGG
===============================
```


## Run
```
    cd examples
    bash run.sh
```
You can change running parameters in the run.sh.
