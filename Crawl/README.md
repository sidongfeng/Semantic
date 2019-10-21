# Dribbble Crawler

This project is a web crawler, aimed to crawling designs from Dribbble associated with meta data including title, author, tags, etc. 

## Installation

1. Ensure Chromium/Google Chrome is installed in a recognized location.

2. Download the [ChromeDriver](https://chromedriver.chromium.org/downloads) binary for your platform (match Chrome version)

3. (Optional) Include the ChromeDriver location in your PATH environment variable


## Preparation

First of all, clone the code
```
git clone ??
```

### Compilation

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

## Usage

```
python Crawler.py
```

Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --chromedriver  	        |./chromedriver/| Chrome Driver Location
| --ipath                   |./images/      | Image Save Location
| --mpath 		            |./Metadata.csv	| Metadata Save Location
| --headless                |   true        | Chrome headless
| -h --help                 |               | show this help message and exit

## Authorship

This project is contributed by [Sidong Feng](https://github.com/u6063820).

## License

TODO: Write license