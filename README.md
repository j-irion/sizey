### Repository to the Sizey Paper.

```bibtex
@inproceedings{bader2024Sizey,
  author={Bader, Jonathan and Skalski, Fabian and Lehmann, Fabian and Scheinert, Dominik and Will, Jonathan and Thamsen, Lauritz and Kao, Odej},
  booktitle={2024 IEEE International Conference on Cluster Computing (CLUSTER)}, 
  title={Sizey: Memory-Efficient Execution of Scientific Workflow Tasks}, 
  year={2024},
}
```

### Run Sizey

1. Create a Python virtual environment and install the dependencies
2. Run `python3 main.py filename alpha softmax error_metric seed [batch_size] [retrain_interval]`

- `filename` describes the workflow from the data folder. For instance `./data/trace_methylseq.csv`  
- `alpha` sets the alpha you want to execute Sizey with. It has to be between 0.0 and 1.0  
- `softmax` toggles the softmax ensemble strategy. Set to `True` to use it, otherwise `False` for the argmax strategy.
- `error_metric` defines the XYZ used for ABC. Currently, it is either `smoothed_mape` or `neg_mean_squared_error` whereas `smoothed_mape` should be used and other error metrics might be experimental and change the impact on the RAQ score.  
- `seed` defines the seed for splitting up the initial data in training and test data and also defines the order of online task input.
- `batch_size` (optional) controls after how many samples the predictors update their models. Default is `1`.
- `retrain_interval` (optional) triggers a full re-training of the incremental models after processing a number of mini-batches. Set to `0` to disable.

Here is an example command: `./data/trace_methylseq.csv 0.0 True smoothed_mape 1996`

- Check the results in terminal, and in results folder.
