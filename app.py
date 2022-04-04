import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


df = pd.read_csv("data/news.csv")
labels = df.label
x_train, x_test, y_train, y_test = train_test_split(
    df["text"], labels, test_size=0.2, random_state=7
)
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
pac = PassiveAggressiveClassifier(max_iter=49)
pac.fit(tfidf_train, y_train)


def predict_fake(text, print_result=False, return_result=True):
    global tfidf_vectorizer
    global pac

    tfidf_text = tfidf_vectorizer.transform([text])

    if print_result == True:
        print(pac.predict(tfidf_text))
    if print_result == True:
        return pac.predict(tfidf_text)


if __name__ == "__main__":
    print("wrong file, buddy")
