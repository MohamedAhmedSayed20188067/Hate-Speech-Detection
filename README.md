# Hate Speech Detection Project 🚫💬

A comprehensive Natural Language Processing (NLP) project for detecting hate speech in text using multiple machine learning approaches including traditional ML algorithms and transformer-based models.

## 📋 Project Overview

This project implements hate speech detection using various machine learning techniques and datasets from different sources. The system can classify text as normal speech or hate speech, with additional capabilities for offensive language detection.

## 🎯 Features

- **Multi-dataset Training**: Utilizes data from movies, Fox News comments, and Twitter
- **Multiple ML Approaches**: Implements XGBoost, Logistic Regression, and BERT-based models
- **Text Preprocessing**: Advanced text cleaning, lemmatization, and feature extraction
- **Interactive Chatbot**: GUI-based chatbot for real-time hate speech detection
- **Data Visualization**: Comprehensive exploratory data analysis with visualizations

## 📊 Datasets

The project uses three main datasets:

1. **Movies Dataset** (`all_movies.csv`)
   - Movie-related comments and reviews
   - Labels: 0 (normal), 1 (offensive), 2 (hate speech)

2. **Fox News Dataset** (`fox_news.csv`)
   - News article comments
   - Binary classification (hate/normal)

3. **Twitter Dataset** (`twitter.csv`)
   - Twitter posts and tweets
   - Binary classification (hate/normal)

## 🔧 Technologies Used

### Libraries & Frameworks
- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Transformers** - BERT model implementation
- **TensorFlow** - Deep learning framework
- **NLTK** - Natural language processing
- **SpaCy** - Advanced NLP processing
- **Matplotlib/Seaborn** - Data visualization
- **Tkinter** - GUI development

### Machine Learning Models
- **XGBoost Classifier**
- **Logistic Regression**
- **BERT (Bidirectional Encoder Representations from Transformers)**

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/MohamedAhmedSayed20188067/Hate-Speech-Detection.git
cd Hate-Speech-Detection
```

2. Install required dependencies:
```bash
pip install pandas numpy scikit-learn xgboost transformers tensorflow nltk spacy matplotlib seaborn
```

3. Download spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

4. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 📁 Project Structure

```
Hate-Speech-Detection/
├── projectNLP.ipynb          # Main project notebook with ML models
├── chatbot.ipynb             # Interactive chatbot implementation
├── data/
│   ├── all_movies.csv        # Movies dataset
│   ├── fox_news.csv          # Fox News comments dataset
│   └── twitter.csv           # Twitter dataset
├── hateSpeachModel/          # Trained BERT model
├── NLP_GROUP7_Project_Proposal.pdf
├── NLP_Project_Report.pdf
├── presentation.pdf
└── README.md
```

## 🚀 Usage

### Running the Main Analysis
Open and run `projectNLP.ipynb` in Jupyter Notebook or any compatible environment:

```bash
jupyter notebook projectNLP.ipynb
```

### Using the Chatbot
Run the chatbot interface from `chatbot.ipynb`:

```bash
jupyter notebook chatbot.ipynb
```

The chatbot provides a GUI where you can:
- Enter text to check for hate speech
- Get real-time predictions
- User input is masked for privacy

## 🔍 Methodology

### 1. Data Preprocessing
- **Text Cleaning**: Removal of punctuation, numbers, and special characters
- **Tokenization**: Breaking text into individual tokens
- **Lemmatization**: Converting words to their base forms
- **Stopword Removal**: Filtering out common words
- **TF-IDF Vectorization**: Converting text to numerical features

### 2. Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Reducing feature dimensions while preserving variance

### 3. Model Training
- **Traditional ML**: XGBoost and Logistic Regression with TF-IDF features
- **Deep Learning**: Fine-tuned BERT model for sequence classification

### 4. Evaluation
- Classification reports with precision, recall, and F1-scores
- Confusion matrices for performance visualization
- Cross-validation for robust model assessment

## 📈 Results

The project implements multiple approaches for comparison:

- **XGBoost**: Fast and efficient gradient boosting
- **Logistic Regression**: Baseline linear classifier
- **BERT**: State-of-the-art transformer model for superior accuracy

Performance metrics include accuracy, precision, recall, and F1-score for each model.

## 🎮 Interactive Features

### Chatbot Interface
- **GUI-based**: User-friendly interface built with Tkinter
- **Real-time Detection**: Instant hate speech classification
- **Privacy Protection**: User input is masked in the interface
- **Binary Classification**: Classifies as 'hate' or 'normal'

## 📚 Academic Context

This project was developed as part of an NLP course (Group 7) and includes:
- Detailed project proposal
- Comprehensive project report
- Presentation materials
- Academic documentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is available for educational and research purposes.

## 👥 Team

**Group 7 - NLP Project**
- Mohamed Ahmed Sayed (ID: 20188067)

## 📞 Contact

For questions or collaborations, please reach out through GitHub issues or contact information provided in the academic documentation.

---

**Note**: This project is designed for educational purposes and research in NLP. The hate speech detection models should be used responsibly and with consideration for ethical implications.