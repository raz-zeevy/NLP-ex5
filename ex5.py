###################################################
# Exercise 5 - Natural Language Processing 67658  #
###################################################
import pandas as pd
import plotly.express as px
import numpy as np

# subset of categories that we will use
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }

def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train',
                                    remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test',
                                   remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion * len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


# Q1
def linear_classification(portion=1.):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    tf = TfidfVectorizer(stop_words='english', max_features=1000)
    # Add your code here
    x_train, y_train, x_test, y_test = get_data(
        categories=category_dict.keys(), portion=portion)
    x_train, x_test = tf.fit_transform(x_train), tf.transform(x_test)
    clf = LogisticRegression(random_state=0).fit(x_train, y_train)
    return np.mean(clf.predict(x_test) == y_test)


# Q2
def transformer_classification(portion=1.):
    """
    Transformer fine-tuning.
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    import torch

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset object
        """
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in
                    self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    from datasets import load_metric
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    from transformers import Trainer, TrainingArguments
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base',
                                              cache_dir=None,
                                              padding='longest',
                                              truncation=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilroberta-base',
        cache_dir=None,
        num_labels=len(category_dict),
        problem_type="single_label_classification")
    x_train, y_train, x_test, y_test = get_data(
        categories=category_dict.keys(), portion=portion)
    # Add your code here
    training_args = TrainingArguments(
        output_dir="/output",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
    )
    # Define the DataLoader
    train_dataset = Dataset(tokenizer(x_train, truncation=True, padding=True), y_train)
    test_dataset = Dataset(tokenizer(x_test, truncation=True, padding=True), y_test)

    # Creating Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics
    )
    trainer.train()
    # see https://huggingface.co/docs/transformers/v4.25.1/en/quicktour#trainer-a-pytorch-optimized-training-loop
    # Use the DataSet object defined above. No need for a DataCollator
    return trainer.evaluate(test_dataset)["eval_accuracy"]


# Q3
def zeroshot_classification(portion=1.):
    """
    Perform zero-shot classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from transformers import pipeline
    from sklearn.metrics import accuracy_score
    import torch
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    clf = pipeline("zero-shot-classification",
                   model='cross-encoder/nli-MiniLM2-L6-H768',
                   device='cuda:0' if torch.cuda.is_available() else 'cpu')
    candidate_labels = list(category_dict.values())

    # Add your code here
    res_dict = {categoty: i for i, categoty in enumerate(candidate_labels)}
    # Add your code here
    y_pred = [clf(x, candidate_labels=candidate_labels) for x in
              x_test]
    y_pred = [res_dict[pred['labels'][np.argmax(pred['scores'])]] for pred in
              y_pred]
    # Calculate the accuracy
    return accuracy_score(y_test, y_pred)


def question_1():
    print("Logistic regression results:")
    accuracies = []
    for p in portions:
        p_acc = linear_classification(p)
        accuracies.append(p_acc)
        print(f"portion {p} accuracy: {p_acc}")
    # Plot the data
    fig = px.line(x=portions, y=accuracies, title="Logistic regression "
                                                  "accuracy vs. portion of "
                                                  "data",
                  labels={'x': 'Portion of data', 'y': 'Accuracy'})
    fig.show()


def question_2():
    accuracy = []
    print("\nFinetuning results:")
    for p in portions:
        p_acc = transformer_classification(p)
        accuracy.append(p_acc)
        print(f"portion {p} Accuracy: {p_acc}")
        print(transformer_classification(portion=p))
    # plot
    fig = px.line(x=portions, y=accuracy, title="Transformers accuracy "
                                                "vs. portion of data",
                  markers=True,
                  labels={'x': 'Portion of data', 'y': 'Accuracy'})
    fig.show()


if __name__ == "__main__":
    portions = [0.1, 0.5, 1.]
    # Q1
    question_1()
    #
    # Q2
    question_2()

    # # Q3
    print("\nZero-shot result:")
    print(zeroshot_classification())
