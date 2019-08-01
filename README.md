# Semantics
## Overview
<!-- ## Overview
Graphical User Interface (GUI) is ubiquitous, but good GUI design is challenging and time-consuming. Despite the enormous amount of UI designs existed online, it is still difficult for designers to efficiently find what they want, due to the gap between the UI design image and textural query. To overcome that problem, design sharing sites like Dribbble ask users to attach tags when uploading tags for searching. However, designers may use different keywords to express the same meaning or miss some keywords for their UI design, resulting in the difficulty of retrieval. This paper introduces an automatic approach to recover the missing tags for the UI, hence finding the missing UIs. Through an iterative open coding of thousands of existing tags, we construct a vocabulary of UI semantics with high-level categories.  -->


## Getting Started
Put files in the following directory structure.

    .
    ├── Data  
    |   ├── images  
    |   ├── platform
    |   ├── table.csv
    |   ├── Metadata.csv
    |   ├── glove.6B.50d.txt (Not included)
    |   ├── to_dataset.py
    |   ├── autoaugment.py
    |   └── categorization.py
    ├── Src
    ├── Ruiqi
    ├── ANN
    ├── Resnet+ANN
    ├── Resnet
    └── Readme.md


### Prerequisites

Package required

```
Name: numpy         Version: 1.16.3
Name: pandas        Version: 0.24.1
Name: torch         Version: 1.0.1.post2
Name: matplotlib    Version: 3.0.2
Name: python        Version: 3.7.2
...
```

File required

```
glove.6B.50d.txt: [here](https://drive.google.com/open?id=1ublNdoeX8i5iTmwP_F-C1jS3SOzcHFT8)
```

### Installing

Use pip to easy install all packages. For example,

```
pip3 install numpy
```


## Running the code
To generate the dataset with tag (default: blue)
```
cd Data
python3 to_dataset.py
```
To train the network with tag (default: blue)
```
cd Src
python3 train.py
```
Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --tag  	                |	blue        | model/tag to train
| --epoch                   |   100         | upper epoch limit
| --lr LR  		            |   1e-4	    | initial learning rate
| --optim OPTIM 	        |   Adam        | optimizer to use
| --seed SEED	            |   42          | random seed
| --patience                |   5           | number of patience to early stop
| --mode                    |   true        | finetune the partial model
| --pretrain                |   true        | use pretrain network
| --lr_decay                |   false       | lr decay
| -h --help                 |               | show this help message and exit

<!-- Supportive code:
- Early stop:   earlystop.py
- BDR:          BDR.py
- General TCN:  tcn.py -->

<!-- ### Data

See `save_series_labels` in `preprocess.py`. You only need to process the data once, or reprocess to 
generate new data. The default path to store the data is at `./data/npy-file`.

Original source of the data can be found [here](http://cs.anu.edu.au/~tom/datasets/thermal-stress_v2.zip).

### Note

- Because the receptive field depends on depth of the network and the filter size, we need
to make sure these the model we use can cover the sequence. 

- While this is a sequence model task, we only use the very last output (i.e. at time T=10) for 
the eventual classification. -->

## Authors

* **Sidong Feng**  - [Gitlab](https://gitlab.cecs.anu.edu.au/u6063820)



<!-- ## Acknowledgments

* Bimodal Distribution Removal
* Tukeys Method 
* SMOTE
* Noise
* Outlier detection
* Outlier treatment
* Neural Network -->