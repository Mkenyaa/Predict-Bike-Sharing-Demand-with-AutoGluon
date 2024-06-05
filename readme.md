# Predict Bike Sharing Demand with AutoGluon


## Overview
In this project, our goal is to leverage Machine Learning Engineering techniques to participate in a Kaggle competition, utilizing the AutoGluon library.

To begin, you'll need to create a Kaggle account if you don't already have one. Then, download the Bike Sharing Demand dataset and proceed to train a model using AutoGluon. Once the model is trained, we will submit the initial results to obtain an initial ranking in the competition.

Following the completion of the first workflow, we'll embark on an iterative process to improve our score further. This will involve incorporating additional features into the dataset and fine-tuning the hyperparameters available with AutoGluon.

Ultimately, we will compile all the work into a report that outlines the methods employed for achieving the best score improvements and the rationale behind their effectiveness. This report will provide insights into the key techniques and strategies that contributed to enhancing the model's performance throughout the project.

To meet specifications, the project will require at least these files:
* Jupyter notebook with code run to completion
* HTML export of the jupyter notebbook
* Markdown or PDF file of the report

Images or additional files needed to make your notebook or report complete can be also added.

## Getting Started
* Clone this template repository `https://github.com/eljandoubi/Predict-Bike-Sharing-Demand-with-AutoGluon.git` into AWS Sagemaker Studio (or local development).

* Proceed with the project within the [jupyter notebook](project-template.ipynb).
* Visit the [Kaggle Bike Sharing Demand Competition](https://www.kaggle.com/c/bike-sharing-demand) page. There you will see the overall details about the competition including overview, data, code, discussion, leaderboard, and rules. You will primarily be focused on the data and ranking sections.

### Dependencies

```
Python 3.7
MXNet 1.8
Pandas >= 1.2.4
AutoGluon 0.2.0
```

### Installation
For this project, it is highly recommended to use Sagemaker Studio from the course provided AWS workspace. This will simplify much of the installation needed to get started.

For local development, you will need to setup a jupyter lab instance.
* Follow the [jupyter install](https://jupyter.org/install.html) link for best practices to install and start a jupyter lab instance.
* If you have a python virtual environment already installed you can just `pip` install it.
```
pip install jupyterlab
```
* There are also docker containers containing jupyter lab from [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html).
