# DIRL: Domain-Invariant Robot Learning

Domain-Invariant Robot Learning (DIRL) is a novel algorithm that semantically aligns both the marginal and the conditional distributions across source and target enviroments.

Follow the instructions to get started:
```
pip install -r requirements.txt
python setup.py develop
```

## Examples

Below we provide a couple of examples in 2D environ to illustrate the efficiency of the DIRL algorithm in learning invariant feature representations. Customize your results by adjusting the settings and hyperparameters in the respective config files in the `configs` folder.
 
### 2D Synthetic Domain

```
python src/train_synthetic_2d.py -model dirl -num_target_labels 8 
```

Results are saved in `results/figs` and `results/animations` folder.

Vary the number of target examples `num_target_examples` and `-mode` among `[source_only, dann, triplet, dirl]`  to analyse  the performance of the algorithm.

### Digits Bechmarks 

```
python src/train_digits.py -mode dirl -source mnist -target mnistm -num_target_labels 10 -save_results True
```
Results are saved in `results` folder.

Vary the `-source` and `-target` options among `[mnist, mnistm, svhn, usps]`. 

`-mode` and `-num_target_labels` can be changed similarly as above.

