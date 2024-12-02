# 🏈 NFL Sentiment Analysis

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![RoBERTa](https://img.shields.io/badge/RoBERTa-Transformer-yellow.svg)](https://huggingface.co/roberta-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A sophisticated machine learning project that performs real-time sentiment analysis on NFL-related social media content using the RoBERTa pre-trained model. The analysis provides valuable insights into public sentiment towards NFL teams during the active season.

📊 [View Dataset on Kaggle](INSERT_KAGGLE_LINK)  
📝 [Read the Article on Medium](INSERT_MEDIUM_LINK)

## 🌟 Key Features

- Real-time social media data collection for NFL teams
- Context-aware sentiment analysis using RoBERTa
- Intelligent parsing of NFL-specific scenarios:
  - Draft pick implications
  - Injury impact assessment
  - Team performance metrics
- Automated data export and visualization
- Season-specific context optimization

## 📊 Examples

![NFL Sentiment Analysis Dashboard](https://github.com/user-attachments/assets/d5f6069e-2952-498a-9bd6-4925c4bcd3e2)

### Model Context Intelligence

The model has been specifically trained to understand NFL-specific contexts:
- Negative sentiment for season-ending injuries
- Draft position implications during active season
- Team performance trends and playoff implications

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/cam-cc/NFL-sentiment-analysis.git
cd NFL-sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Configure your settings:
```python
# create a .env
TWITTER_USERNAME=Username
TWITTER_PASSWORD=Password
```

2. Run the analysis:
   **FROM ROOT DIR**
```bash
python ./src/main.py
```

3. Test custom inputs:
```bash
python ./tests/test-roberta.py
```

Data will be exported to `data/TEAM_TIMESTAMP.csv`

## 📈 Model Testing

To test the sentiment analysis on custom text:

1. Open `test-roberta.py`
2. Modify the sample text
3. Run the script
4. View the comprehensive sentiment analysis output

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✨ Acknowledgments

- RoBERTa model developers
- Twitter I guess

## ⚠️ Note

This model is optimized for in-season analysis. Performance may vary during the off-season due to different contextual indicators.
