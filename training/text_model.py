import pandas as pd
from embetter.text import SentenceEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from embetter.text import SentenceEncoder
from setfit import SetFitTrainer, SetFitModel
from sentence_transformers.losses import CosineSimilarityLoss
from transformers import AutoTokenizer
import joblib
from datasets import load_dataset
from datasets import Dataset
import numpy as np
import os

class TextClassifier:
    def __init__(self, max_iter=5000, sklearn=True):
        if sklearn:
            self.model = make_pipeline(
                SentenceEncoder(),
                LogisticRegression(max_iter=max_iter)
            )
        else:
            self.model = None

    def train_header(self, texts, labels, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(y_pred)
        print(y_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Training completed. Accuracy: {accuracy:.4f}")

    def train_backbone(self, texts, labels, test_size=0.2, random_state=42, num_epochs=5, num_iterations=500):
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=random_state)
        train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
        eval_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})

        backbone = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        trainer = SetFitTrainer(
            model=backbone,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss_class=CosineSimilarityLoss,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
        )
        trainer.train()
        metrics = trainer.evaluate()
        print(metrics)
        self.model = trainer.model

        # self.model.named_steps['sentenceencoder'] = trainer.model.model_body
        # self.model.named_steps['logisticregression'] = trainer.model.model_head

        # evaluation
        embeddings = trainer.model.encode(X_test, batch_size=32, show_progress_bar=None)
        probs = trainer.model.model_head.predict_proba(embeddings)
        print("in backbone model:")
        print(y_test)
        print(probs)


    def save_model(self, file_path, sklearn=True):
        if sklearn:
            joblib.dump(self.model, file_path+".pkl")
        else:
            self.model._save_pretrained(file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, sklearn=True):
        if sklearn:
            self.model = joblib.load(file_path+".pkl")
        else:
            self.model = SetFitModel.from_pretrained(file_path)
        print(f"Model loaded from {file_path}")

    def predict(self, texts):
        return self.model.predict(texts)
    
    def predict_proba(self, texts, sklearn=True):
        if sklearn:
            return self.model.predict_proba(texts)
        else:
            embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=None)
            return self.model.model_head.predict_proba(embeddings)

# Example usage
if __name__ == "__main__":
    folder_path = 'training/data'
    model_path = 'training/models'
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            output_path = os.path.join(model_path, f"model_{os.path.splitext(file_name)[0]}")

            # Load the dataset from the CSV file
            dataset = load_dataset('csv', data_files=file_path)

            # Initialize the classifier
            classifier = TextClassifier(sklearn=False)

            # Train the model
            classifier.train_backbone(dataset["train"]['text'], dataset["train"]['label'])
            # classifier.train_header(dataset["train"]['text'], dataset["train"]['label'])

            # Save the model
            classifier.save_model(output_path, sklearn=False)

            # Load the model
            classifier.load_model(output_path, sklearn=False)

            # Predict
            texts_to_predict = [
                'Language is essentially a complex, intricate system of human expressions governed by grammatical rules. It poses a significant challenge to develop capable AI algorithms for comprehending and grasping a language. As a major approach, language modeling has been widely studied for language understanding and generation in the past two decades, evolving from statistical language models to neural language models. Recently, pre-trained language models (PLMs) have been proposed by pre-training Transformer models over large-scale corpora, showing strong capabilities in solving various NLP tasks. Since researchers have found that model scaling can lead to performance improvement, they further study the scaling effect by increasing the model size to an even larger size. Interestingly, when the parameter scale exceeds a certain level, these enlarged language models not only achieve a significant performance improvement but also show some special abilities that are not present in small-scale language models. To discriminate the difference in parameter scale, the research community has coined the term large language models (LLM) for the PLMs of significant size. Recently, the research on LLMs has been largely advanced by both academia and industry, and a remarkable progress is the launch of ChatGPT, which has attracted widespread attention from society. The technical evolution of LLMs has been making an important impact on the entire AI community, which would revolutionize the way how we develop and use AI algorithms. In this survey, we review the recent advances of LLMs by introducing the background, key findings, and mainstream techniques. In particular, we focus on four major aspects of LLMs, namely pre-training, adaptation tuning, utilization, and capacity evaluation. Besides, we also summarize the available resources for developing LLMs and discuss the remaining issues for future directions.',
            ]
            predictions = classifier.predict_proba([texts_to_predict[0]], sklearn=False)
            print(f"Predictions: {predictions}")
