# 🔍 TF-IDF Implementation for RAG Systems

A comprehensive implementation of Term Frequency-Inverse Document Frequency (TF-IDF) algorithms for understanding text vectorization in Retrieval-Augmented Generation (RAG) systems.

[**🧠 Mastering RAG: Empowering AI Models with Our Custom Data**][rag-blog]

[rag-blog]: https://aicodegeek.com/2025/06/29/RAG-Empowering-AI-Models-with-Our-Custom-Data

[**🚀 Complete Guide to Vectors and TF-IDF in RAG Systems**][rag-blog2]

[rag-blog2]: https://aicodegeek.com/2025/06/29/Complete-Guide-tt-Vectors-and-TF-IDF-in-RAG-Systems

## 📚 Overview

This project demonstrates how to convert text documents into numerical vectors using TF-IDF, which forms the foundation for understanding modern embedding techniques used in RAG systems. The implementation includes both manual calculations and scikit-learn integration.

## 🎯 What You'll Learn

- **TF-IDF Fundamentals**: Understanding how text becomes searchable vectors
- **Manual Implementation**: Step-by-step TF-IDF calculation from scratch
- **Production Ready**: Using scikit-learn for optimized vectorization
- **Document Similarity**: Calculating cosine similarity between documents
- **RAG Applications**: How this applies to modern AI retrieval systems

## 📁 Project Structure

```
tf-idf-project/
├── tf-idf-calculation.py      # Basic TF-IDF calculator with detailed analysis
├── tf-idf-vectorizer.py       # Advanced vectorization with multiple approaches
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project files**
   ```bash
   # If using git
   git clone <your-repo-url>
   cd tf-idf-project
   
   # Or simply download the Python files to a folder
   ```

2. **Install required packages**
   ```bash
   pip install numpy scikit-learn
   ```

   Or install all at once:
   ```bash
   pip install -r requirements.txt
   ```

   **If you get permission errors, try:**
   ```bash
   pip install --user numpy scikit-learn
   ```

3. **Run the basic TF-IDF calculator**
   ```bash
   python tf-idf-calculation.py
   ```

4. **Run the advanced vectorizer**
   ```bash
   python tf-idf-vectorizer.py
   ```

## 🔧 Package Installation Guide

### For Different Operating Systems

#### Windows
```cmd
pip install numpy scikit-learn
```

#### macOS/Linux
```bash
pip3 install numpy scikit-learn
```

#### Using Anaconda/Miniconda
```bash
conda install numpy scikit-learn
```

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv tf-idf-env

# Activate it
# On Windows:
tf-idf-env\Scripts\activate
# On macOS/Linux:
source tf-idf-env/bin/activate

# Install packages
pip install numpy scikit-learn

# When done, deactivate
deactivate
```

## 📖 Usage Examples

### Basic TF-IDF Analysis

```python
from tf_idf_calculation import TFIDFCalculator

# Your documents
documents = [
    "Python is a programming language",
    "Java is also a programming language", 
    "Both Python and Java are popular"
]

# Create calculator
calculator = TFIDFCalculator(documents)

# Analyze a term
calculator.print_analysis("Python")
```

### Advanced Vectorization

```python
from tf_idf_vectorizer import TextToVectorConverter

# Initialize converter
converter = TextToVectorConverter(documents)

# Create different types of vectors
tfidf_vectors = converter.create_tfidf_vectors_manual()
sklearn_vectors, features = converter.create_tfidf_vectors_sklearn()
binary_vectors = converter.create_binary_vectors()

# Analyze document similarity
converter.analyze_document_similarity(tfidf_vectors)
```

## 🔍 What Each Script Does

### 📊 tf-idf-calculation.py

**Purpose**: Demonstrates step-by-step TF-IDF calculation

**Features**:
- Manual TF (Term Frequency) calculation
- Manual IDF (Inverse Document Frequency) calculation
- Detailed analysis output for understanding
- Document ranking by relevance

**Sample Output**:
```
=== TF-IDF Analysis for 'Python' ===

IDF(Python) = 0.405465

Document 1:
  Content: "Python is a high-level, interpreted programming language..."
  Total words: 13
  Term 'Python' appears: 1 times
  TF(Python) = 1/13 = 0.076923
  TF-IDF(Python) = 0.076923 × 0.405465 = 0.031190
```

### 🚀 tf-idf-vectorizer.py

**Purpose**: Production-ready vectorization with multiple approaches

**Features**:
- Manual TF-IDF vector creation
- Scikit-learn TF-IDF implementation
- Binary and count vectors
- Cosine similarity calculation
- Word embeddings example
- Document similarity analysis

**Sample Output**:
```
=== Manual TF-IDF Vectors ===

Document 1: Python is a high-level, interpreted programming...
Vector shape: (29,)
Top features:
  python: 0.031190
  interpreted: 0.025342
  simplicity: 0.025342
```

## 🎓 Understanding the Output

### TF-IDF Scores
- **Higher scores** = More relevant to the term
- **0.0 scores** = Term doesn't appear in document
- **Comparative scores** help rank document relevance

### Document Similarity
- **1.0** = Identical documents
- **0.0** = Completely different documents
- **0.3-0.7** = Moderate similarity
- **0.8+** = High similarity

### Vector Shapes
- **Manual vectors**: Size equals vocabulary size
- **Sklearn vectors**: Optimized size based on parameters
- **Dense vs Sparse**: Most vectors are sparse (many zeros)

## 🛠️ Troubleshooting

### Common Issues

#### ImportError: No module named 'numpy'
```bash
pip install numpy
```

#### ImportError: No module named 'sklearn'
```bash
pip install scikit-learn
```

#### Permission denied errors
```bash
pip install --user numpy scikit-learn
```

#### Python version issues
```bash
python3 -m pip install numpy scikit-learn
```

### Environment Issues

#### Check Python version
```bash
python --version
# Should be 3.7+
```

#### Check installed packages
```bash
pip list | grep -E "(numpy|scikit)"
```

#### Verify installation
```python
import numpy
import sklearn
print("All packages imported successfully!")
```

## 🔗 Connection to RAG Systems

This TF-IDF implementation helps you understand:

1. **Text Vectorization**: How documents become searchable vectors
2. **Similarity Calculation**: How systems find relevant content
3. **Ranking Algorithms**: How results are ordered by relevance
4. **Foundation Knowledge**: Basis for understanding modern embeddings

### Progression Path

```
TF-IDF (This Project) → Word2Vec → BERT → Modern RAG Embeddings
```

## 📈 Next Steps

After mastering this implementation:

1. **Experiment** with different document sets
2. **Modify parameters** in TfidfVectorizer
3. **Compare results** between manual and sklearn implementations
4. **Try sentence transformers** for semantic embeddings
5. **Build a simple RAG system** using these concepts

## 📚 Additional Resources

- **Blog Post**: [RAG: Empowering AI Models with Our Custom Data](https://aicodegeek.com/2025/06/29/RAG-Empowering-AI-Models-with-Our-Custom-Data)
- **Scikit-learn Documentation**: [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- **NumPy Documentation**: [numpy.org](https://numpy.org/doc/)

## 🤝 Contributing

Feel free to:
- Add more document examples
- Implement additional vectorization methods
- Optimize performance
- Add visualization features
- Extend for multilingual support

## 📄 License

This project is provided for educational purposes. Feel free to use and modify for learning and non-commercial projects.

---

**Happy Learning! 🚀**

For questions or issues, please refer to the troubleshooting section or check the blog post for detailed explanations.