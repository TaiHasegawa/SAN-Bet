# SAN-Bet
This repository provides a PyTorch implementation of SAN-Bet.

### Package Requirements
Experiments were conducted with packages below:
* Python 3.6.5
* NumPy 1.19.2
* SciPy 1.5.4
* Pytorch 1.6.0
* NetworkX 2.5
* NetworKit 7.1

### Ranking approximation
For ranking approximation on scale-free graph, run the code as:  
`python run_Ranking.py --g SF`  
"SF" can be replaced by "ER" or "GRP".

### Top-K% Identification
For Top-1% Identification on scale-free graph, run the code as:  
`python run_TopK.py --g SF --k 0.01`  
"SF" can be replaced by "ER" or "GRP", and 0 < k < 1.
