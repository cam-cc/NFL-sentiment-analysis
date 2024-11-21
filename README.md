# NFL Sentiment Analysis
This repository contains a Python project that scrapes Twitter data and performs sentiment analysis on tweets related to NFL teams using the RoBERTa pre-trained model.

## Key Features

- Scrapes tweets related to NFL teams
- Analyzes the sentiment of the tweets using the RoBERTa model
- Provides insights into the sentiment towards different NFL teams

## Examples
![image](https://github.com/user-attachments/assets/d5f6069e-2952-498a-9bd6-4925c4bcd3e2)

## Installation

1. Clone the repo
```
git clone https://github.com/cam-cc/NFL-sentiment-analysis.git
```

2. install necessary requirements
```
pip install -r requirements.txt
```

## Usage

3. Update the necessary configuration in the config.py file.
Run the main script:
```
python main.py
```
Data will be saved to data/TEAM_TIME.csv

4. to test the model modify some of the text in the **test-roberta.py** file and run it, you will get comprehensive analytics regarding the tweet/sentiment
  showcased in the Examples section
## Contributing
If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
