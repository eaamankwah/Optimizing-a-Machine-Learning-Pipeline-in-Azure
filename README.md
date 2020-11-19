# Optimizing-a-Machine-Learning-Pipeline-in-Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.n this project, I build and optimize an Azure ML pipeline 
using the Python SDK and hyperdrive optimization and a provided custom Scikit-learn model. This model is then compared to an Azure AutoML run.

The figure below shows the main steps in creating and optimizing a machine learning pipeline:
![Main Steps](https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/creating-and-optimizing-an-ml-pipeline.png)


## Summary
The dataset for this project was obtained from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).
The dataset contained data about a bank telemarking campaign in Portuguese. The target was to do a binary classification indicating 
whether an individual will sign up for a term deposit or not.

As indicated in the overview section, two approaches where adapted. Firstly, the sklearn logistic regression model was hyperparameter
tuned by using Azure machine learning [HyperDrive package](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?preserve-view=true&view=azure-ml-py) to obtained the best accuracy matric. Secondarily, the model was automatically 
tuned for the best hyperparameters by using Azure AutoML. These processes automate the time consuming and iterative process of the 
development of machine learning models.

The Hyperdrive run achieved an accuracy of **0.9099** with hyperparameters: 
['--C', '47.23308871724378', '--max_iter', '125'] while the AutoML run achieved a better accuracy score of **0.9166**.

## Scikit-learn Pipeline
### The Architecture
The provided training script (train.py) contains a url which downloads the dataset using the TabularDatasetFactory Class object. The training script also encodes a data cleaning function (clean_data) which preprocesses the data and outputs two pandas data frames as the feature sets and the target. The dataset was split into train and test at a ratio of 80:20.
The classification algorithm, [LogisticRegression estimator](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) from [scikit learn library]( https://scikit-learn.org/stable/index.html) is optimized by using the Azure ML Hyperdrive package. Accuracy was used as the primary matric for this classification problem. 

The main logistic regression hyperparameters that were optimized were the inverse regularization (C) float value and the maximum 
iterations (max_iter) integer value. The numerical value “C ” defines the inverse regularization strength that helps to prevent model 
overfitting. The smaller the C value the stronger the regularization. The 'max_iter' defines the maximum number of iterations taken 
for the solvers to converge.

The second hyperparameter is called 'max_iter' which is a numerical value of int. 'max_iter' is the maximum number of iterations taken for the solvers to converge. The default value for 'max_iter' is 100.
The main steps in the pipeline with HyperDrive hyperparameter tuning included the following:

*    Defining the parameter search space as following:

1.    [RandomParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py)  was used to define a random sampling over the hyperparameter search space.

2.    [uniform](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py#uniform-min-value--max-value-)  values were used to acquire 'C' values that were included in the hyperparameter tuning. Uniform defines a uniform distribution from which samples are taken.

3.    [choice](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py#choice--options-) was used to acquire the 'max_iter' values that were included in the  hyperparameter tuning. Choice specifies a discrete set of options to sample from.

*    Accuracy was used as the primary metric to be optimized.

*   Specifying early termination policy for poor performing experiment runs

    *    [BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py) with evaluation_interval and slack_factor was defined as follows:

1.   Bandit policy defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation.

2.    evaluation_interval is the frequency and delay for applying the policy.

3.    slack_factor is the amount of slack allowed with respect to the best performing training run.

*    Resource Allocation

*    [A compute cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python#what-is-a-compute-cluster) with vm_size='STANDARD_D2_V2', and max_nodes=4 was allocated

*    An experiment was launched with the defined configuration
*    The training  runs were visualized
*    The best configuration was selected for the defined logistic regression model.
*    A Workspace and an experiment were initialized prior to the HyperDrive configuration

### The benefits of my parameter sampler

A RandomParameterSampling technique was used to define a random sampling space over uniform values for the inverse regularization parameter and choice-based integer values for the maximum iterations. These set of hyperparameters were randomly selected over combination of  the “C” and “max_iter” values, fitted on the trained data. The benefits include:

*    The efficiency of automation far outweighs the manual and time-consuming iterative processes of hyperparameter tuning.
*   The ability to sample both discrete and continues values for the hyperparameters in the same experiment.
*   The ability to perform automated hyperparameter tuning and running experiments in parallel to efficiently select best parameters.
*    The ability to efficiently terminate poor performing experimental runs.
*    The method usually performs better than grid search method where every combination of parameters is iteratively tested.

### The benefits of the early stopping policy

The Bandit early termination policy was used in this project. Based on the slack factor, and frequency and delay evaluation_interval values selected, the HyperDrive Bandit will terminate poor performing experimental runs. An evaluation_interval and slack_factor of 4 and 0.1, respectively were used indicating that the best performing run score after the fourth interval was compared to the  score of the next run ahead. The run was terminated when the difference between the two scores was smaller than the slack factor. This process helps imminently to improve computational efficiency.

### HyperDrive model details and visualization

![hd1](https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/hd1.png)

![hd2](https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/hd2.png)

![hd4](https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/hd4.png)

![hd5](https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/hd5.png)

![hd6](https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/hd6.png)

## AutoML
![Azure ML in image]( https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/ml_in_one_image.png)

![AutoML concept](https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/automl-concept.png)

The process of machine learning model development is automated in AutoML. This process explores a wide range of models, including RandomForest, StackEnsemble, XgBoost, LightBGM, SGD Classifier and etc. to efficiently train and select the best model. It uses cross validation splits to reduce model overfitting. The best model selected was the VotingEnsemble model which combined the predictions from other multiple classification models.

### AutoML model details and visualization

![aml1]( https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/aml1.png)

![aml2]( https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/aml2.png)

![aml3]( https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/aml3.png)

![aml4]( https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/aml4.png)

![aml5]( https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/aml5.png)

![aml8]( https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/aml8.png)

![aml10]( https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/aml10.png)

## Pipeline comparison
*    The accuracy of AutoML (0.9166) is higher than the accuracy of HyperDrive (0.9099) by **0.0067 (0.67%) **. 
*    The HyperDrive used a simple logistic regression architecture which is easy to configure while AutoML used more complex parameter configuration. 
*   AutoML has the advantage of setting and comparing several models in parallel with different hyperparameter settings while HyperDrive will train one model at the time.
*    HyperDrive model required the setting up of the provided train.py script where the logistic regression was calculated over the weighted sum of the input passed via the sigmoid activation function. AutoML did not use the train.py script.
*    HyperDrive is resource intensive and can be manual while AutoML is efficient, when setting up multiple estimators.
*    In general, AutoML is better than HyperDrive as it allows the selection of the best model among several models running in parallel and allows the Data Scientist to focus more on business problems.

## Future work
### Areas of improvement: HyperDrive

*    Try different combination of  hyperparameter values  --C and --Max_iter. The C value could be selected by using the Uniform or Choice functions to define the search space.
*    Define the parameter search space as discrete or continuous, and a sampling method over the search space as grid, or Bayesian.
*   Try new HyperDrive runs with difference estimators including the best performing estimator VotingEnsemble to improve the accuracy score.
*    Balance the dataset using sampling techniques such as oversampling the minority class or undersampling the majority class before the model optimization step or try an appropriate matrix such as AUC_weighted. The AUC method is not affected by the imbalance in the dataset. 

### Areas of improvement: AutoML

*   Include deep learning in a new AutoML experiment run. Deep learning models can improve the accuracy but may be hard to interpret.
*   Reduce the feature sets in a new AutoML run based the feature importance visualization derived from the earlier model to improve performance.
*   Try BayesianSampling technique which requires no early stopping policy. Since the dataset is not “big”, tuning the hyperparameter to a full completion may improve the accuracy score.
*    Like the hyperdrive run, using an AUC_weighted metric or balancing the dataset may improve the accuracy results

Below is the supported models for AutoML:

![AutoMl supported models](https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/automl_supported_models.png)

## Proof of cluster clean up

![Cluster cleanup]( https://github.com/eaamankwah/Optimizing-a-Machine-Learning-Pipeline-in-Azure/blob/main/screenshots/compute_cleanup.png)

## References

* [Azure Machine Learning Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)

* [Azure Machine Learning SKD](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)

* [Udacity Q & A Platform](https://knowledge.udacity.com/?nanodegree=nd00333&page=1&project=766&rubric=2925&sort=SCORE)
