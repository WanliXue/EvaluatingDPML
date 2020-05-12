# Analysing the Leaky Cauldron

The goal of this project is to evaluates the privacy leakage of differential private machine learning algorithms.

The code has been adapted from the code base (https://github.com/csong27/membership-inference) of membership inference attack work by Shokri et al. (https://ieeexplore.ieee.org/document/7958568).

### Requirements

- Python 2.7 or higher (https://www.anaconda.com/distribution/)
- Tensorflow (https://www.tensorflow.org/install)
- Tensorflow Privacy (https://github.com/tensorflow/privacy)

### Pre-processing data sets

Pre-processed CIFAR-100 data set has been provided in the `dataset/` folder. Purchase-100 data set can be downloaded from Kaggle web site (https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data). This can be pre-processed using the preprocess_purchase.py scipt provided in the repository.
For pre-processing other data sets, bound the L2 norm of each record to 1 and pickle the features and labels separately into `$dataset`_feature.p and `$dataset`_labels.p files in the `dataset/` folder (where `$dataset` is a placeholder for the data set file name, e.g. for Purchase-100 data set, `$dataset` will be purchase_100).


## Evaluating Differentially Private Machine Learning in Practice

Follow the instructions below to replicate the results from the paper *Evaluating Differentially Private Machine Learning in Practice* (https://arxiv.org/abs/1902.08874).

### Training the non-private baseline models for CIFAR

When you are running the code on a data set for the first time, run `python evaluating_dpml.py $dataset --save_data=1` on terminal. This will split the data set into random subsets for training and testing of target, shadow and attack models.

Run `python evaluating_dpml.py $dataset --target_model=$model --target_l2_ratio=$lambda` on terminal.

For training optimal non-private baseline neural network on CIFAR-100 data set, we set `$dataset`='cifar_100', `$model`='nn' and `$lambda`=1e-4. For logsitic regression model, we set `$dataset`='cifar_100', `$model`='softmax' and `$lambda`=1e-5.

For training optimal non-private baseline neural network on Purchase-100 data set, we set `$dataset`='purchase_100', `$model`='nn' and `$lambda`=1e-8. For logsitic regression model, we set `$dataset`='cifar_100', `$model`='softmax' and `$lambda`=1e-5.

### Training the differential private models

Run `python evaluating_dpml.py $dataset --target_model=$model --target_l2_ratio=$lambda --target_privacy='grad_pert' --target_dp=$dp --target_epsilon=$epsilon` on terminal. Where `$dp` can be set to 'dp' for naive composition, 'adv_cmp' for advanced composition, 'zcdp' for zero concentrated DP and 'rdp' for Renyi DP. `$epsilon` controls the privacy budget parameter. Refer to __main__ block of attack.py for other command-line arguments.

### Plotting the results from the paper 

Update the `$lambda` variables accordingly and run `./evaluating_dpml_run.sh $dataset` on terminal. Results will be stored in `results/$dataset` folder.

Run `evaluating_dpml_interpret_results.py $dataset --model=$model --l2_ratio=$lambda` to obtain the plots and tabular results. Other command-line arguments are as follows: 
- `--function` prints the plots if set to 1 (default), or gives the membership revelation results if set to 2.
- `--plot` specifies the type of plot to be printed
    - 'acc' prints the accuracy loss comparison plot (default)
    - 'attack' prints the privacy leakage due to Shokri et al. membership inference attack
    - 'mem' prints the privacy leakage due to Yu et al. membership inference attack
    - 'attr' prints the privacy leakage due to Yu et al. attribute inference attack
- `--silent` specifies if the plot values are to be displayed (0) or not (1 - default)
- `--fpr_threshold` sets the False Positive Rate threshold (refer the paper)


## Revisiting Membership Inference under Realistic Assumptions

To replicate the results of the paper *Revisiting Membership Inference under Realistic Assumptions*, use the same commands as above but replace `evaluating_dpml` with `improved_mi`. For instance, to run the batch file, run `./improved_mi_run.sh $dataset` on terminal.
