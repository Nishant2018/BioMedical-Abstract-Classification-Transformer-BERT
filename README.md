
---

# Biomedical Abstract Classification

This repository contains code for a Transformer-based model to classify biomedical abstracts into different categories. The model is trained on a dataset of biomedical research abstracts and fine-tuned to perform sentence classification.

## Overview

Biomedical abstracts often contain crucial information about research findings, methods, and conclusions. Automated classification of these abstracts can aid in organizing and summarizing large volumes of research data. In this project, we leverage state-of-the-art Transformer architecture to classify biomedical abstracts into predefined categories.

## Key Features

- **Data Preprocessing:** The abstracts undergo extensive preprocessing to remove noise and improve model performance.
- **Fine-Tuning BERT:** We fine-tune a pre-trained BERT model for sentence classification on our dataset.
- **Evaluation Metrics:** We evaluate the model using accuracy, precision, recall, and F1 score.
- **Confusion Matrix Analysis:** Visualize the model's performance using confusion matrices.
- **Experimentation:** Compare the performance of different preprocessing techniques and model configurations.

## Dataset

The dataset used in this project consists of biomedical abstracts categorized into classes such as BACKGROUND, CONCLUSIONS, METHODS, OBJECTIVE, and RESULTS.

## Getting Started

### Prerequisites

- Python 3
- pip

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/biomedical-abstract-classification.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Preprocess the data:
   ```bash
   python preprocess.py
   ```

2. Train the model:
   ```bash
   python train.py
   ```

3. Evaluate the model:
   ```bash
   python evaluate.py
   ```

### Example

```python
# Load the preprocessed data
train_data = pd.read_csv("train_data.csv")

# Train the model
model = BiomedicalTransformer(num_classes)
train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=3)

# Evaluate the model
test_predictions, test_accuracy, test_precision, test_recall, test_f1, test_confusion_mat = evaluate_model(model, test_loader, device)
```

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

## Contact

For any inquiries, please contact [nishantraghuwanshi2018@gmail.com].

--- 
