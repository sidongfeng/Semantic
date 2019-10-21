# UI Tag Semantics

Despite the enormous amount of UI designs existed online, it is still difficult for designers to efficiently find what they want, due to the gap between the UI design image and the textural query. To overcome that problem, design sharing sites like Dribbble ask users to attach tags when uploading tags for searching. However, designers may use different keywords to express the same meaning or miss some keywords for their UI design, resulting in the difficulty of retrieval. This project introduces an automatic approach to recover the missing tags for the UI, hence finding the missing UIs. We create a large-scale UI design dataset with semantic annotations. Through an iterative open coding of thousands of existing tags, we construct a vocabulary of UI semantics with high-level categories. Based on the vocabulary, we train a fusion model on image and tag that recommends the missing tags of the UI design with 82.72% accuracy. 

<div style="color:#0000FF" align="center">
<img src="figures/communitydetection.png"/> 
</div>

<!-- ![UI-related tags association graph](/figures/communitydetection.png) -->

## Preparation

First of all, clone the code
```
git clone https://github.com/u6063820/Semantic
```

### Compilation

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

## Usage

* Crawl designs from Dribbble
```
cd Crawl
```
Please follow the instructions in Crawl to prepare setups. Actually, you can refer to any others. After downloading the data, create softlinks in the folder ./Semantics/ and ./RecoverTags/Data/.

* Distill semantics of UI tags
```
cd Semantics
```
Please follow the instructions in Semantics.

* Predict missing tags
```
cd RecoverTags
```
Please follow the instructions in RecoverTags.

## Authorship

This project is contributed by [Sidong Feng](https://github.com/u6063820).

<!-- ## License

TODO: Write license -->