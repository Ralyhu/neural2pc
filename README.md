# Neural2PC

## Overview

This project is developed as part of the following research paper:

F. Gullo, D. Mandaglio, A. Tagarelli (2024). *Neural Discovery of Balance-aware Polarized Communities* published in Machine Learning (MACH), 2024

Please cite the above paper in any research publication you may produce using this code or data/analysis derived from it.

## Folders
- datasets: contains the real-world and syntethic signed graphs used in the paper.
- code: contains the project code
- output: stores all results produced by the selected algorithm

## Usage examples

From the folder 'neural2pc/code', run one of the following command:
```bash      

run_sgcn.py [-h] -d D [-e E] [-s S] [-i I] [-p P] [-lr LR] [-wd WD] [-gamma GAMMA] [-dr DR] [-l L] [-agg {add,mean,max}]

run_sgdnet.py [-h] -d D [-e E] [-s S] [-i I] [-p P] [-lr LR] [-wd WD] [-gamma GAMMA] [-dr DR] [-l L] [-K K] [-agg {add,mean,max}]

```
### Dependencies
- dotmap==1.3.30
- loguru==0.6.0
- numpy==1.22.1
- pandas==1.4.0
- scikit_learn==1.0.2
- scipy==1.7.3
- torch==1.10.1
- torch_geometric==2.0.3
- torch_sparse==0.6.12
- tqdm==4.62.3

### Positional arguments
```
  -d DATASET, --dataset DATASET
                        Input dataset, whose name identifies a particular file in 'datasets/'
```
### Optional arguments
```                                          
  -e E                 Number of epochs (default 200)
  -s S                 Random seed (default 100)
  -i I                 Iteration number ID (default 0)
  -p P                 Percentage of last epochs to consider for average statistics (default 10)
  -lr LR               Learning rate (default 0.001)
  -wd WD               Weight Decay (L2 regularization) (default 0.01)
  -gamma GAMMA         Gamma weight for polarity (default 1.0)
  -dr DR               Discrete regularizer (default False)
  -l L                 Number of layers (default 2)
  -agg {add,mean,max}  Aggregation operator (default 'mean')
                    
```

## Output format

- `best epoch.txt`: the epoch number that yielded the solution with the highest polarity score.
- `membership.txt`: community assignments for each node (row), where -1 means neutral node, and 1 and 2 correspond to S1 and S2, respectively.
- `edges_info.txt`: the number of (positive/negative) intra/inter-community edges of the discovered pair of polarized communities (the one with the highest polarity found across epochs).
- `last_edges_info.txt`: the average (across the last p\% epochs) number of (positive/negative) intra/inter-community edges of the discovered pair of polarized communities.
- `last_ag_ratio.txt`: average (across the last p\% epochs) agreement ratio of the discovered pair of polarized communities.
- `polarity.txt`: polarity score of the best solution (pair of polarized communities) found across all epochs.
- `last_polarity.txt`: average (across the last p\% epochs) polarity score of the solution (pair of polarized communities).
- `size.txt`: the size of the discovered pair of polarized communities (the one with the highest polarity found across epochs).
- `last_size.txt`: the average (across the last p\% epochs) size of the discovered pair of polarized communities.
- `thresholds.txt`: for the continuous solution (yielded at each epoch, row), it stores (comma-separated values) the average assignment score (absolute value), the number of distinct assignment scores in the solution, the minimum and maximum assignment scores (absolute value), and the threshold \tau which yielded the best (in terms of polarity) discretized solution after rounding.
- `neg_thresholds.txt`: for the continuous solution (yielded at each epoch, row), it stores (comma-separated values) the average negative assignment score, the number of distinct negative assignment scores in the solution, and the minimum and maximum negative assignment scores.
- `pos_thresholds.txt`: for the continuous solution (yielded at each epoch, row), it stores (comma-separated values) the average positive assignment score, the number of distinct positive assignment scores in the solution, and the minimum and maximum positive assignment scores.
- `training_loss.txt`: polarity loss of the continuous solution for each learning epoch (in separate rows).
- `test_loss.txt`: polarity loss of the discretized solution for each learning epoch (in separate rows).
- `training_time.txt`: total execution time of the training steps (across all epochs) of Neural2PC.
- `test_time.txt`: total execution time of the rounding steps (across all epochs) of Neural2PC.

## Contact

If you have any questions or need further assistance, please feel free to contact me at d.mandaglio@dimes.unical.it