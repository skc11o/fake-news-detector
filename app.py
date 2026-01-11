from flask import Flask, request, render_template
import joblib
from sklearn.exceptions import NotFittedError
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from newspaper import Article   
import numpy as np

app = Flask(__name__)

# Load classical models
tfidf = joblib.load("models_final/tfidf_vectorizer.pkl")
nb_model = joblib.load("models_final/nb_tfidf.pkl")
lr_model = joblib.load("models_final/logreg_tfidf.pkl")
svm_model = joblib.load("models_final/svm_tfidf.pkl")

# # Load BERT model
# bert_tokenizer = BertTokenizer.from_pretrained("bert_model")
# bert_model = BertForSequenceClassification.from_pretrained("bert_model")


def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        raise RuntimeError(f"Could not extract article: {str(e)}")


# prediction function
def predict_models(text: str):
    if not text or not text.strip():
        return None

    try:
        X_tfidf = tfidf.transform([text])   
    except NotFittedError:
        raise RuntimeError("TF-IDF vectorizer is not fitted. Re-export the fitted vectorizer from training.")

    def label_from(pred):
        return "Real" if int(pred) == 1 else "Fake"

    def get_probability(model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[0][1]  # probability Real
        elif hasattr(model, "decision_function"):
            score = model.decision_function(X)[0]
            return float(1 / (1 + np.exp(-score)))
        else:
            return 0.5

    results = {}
    for name, model in [
        ("Naive Bayes", nb_model),
        ("Logistic Regression", lr_model),
        ("SVM", svm_model),
    ]:
        pred = model.predict(X_tfidf)[0]
        prob = get_probability(model, X_tfidf)
        label = label_from(pred)

        results[name] = {
            "label": label,
            "prob": prob
        }

    # # BERT
    # inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # with torch.no_grad():
    #     outputs = bert_model(**inputs)
    #     probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    #     pred = int(np.argmax(probs))
    #     label = "Real" if pred == 1 else "Fake"

    #     results["BERT"] = {
    #         "label": label,
    #         "prob": float(probs[1])
    #     }

    return results


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_text = ""
    error = None
    url_input = ""

    if request.method == "POST":
        url_input = request.form.get("url", "")
        input_text = request.form.get("news", "")

        try:
            if url_input.strip():
                input_text = extract_text_from_url(url_input)

            if input_text.strip():
                prediction = predict_models(input_text)
            else:
                error = "Please paste some text or enter a valid URL before clicking Predict."
        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        input_text=input_text,
        url_input=url_input,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
