import re
import random
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------- 1. Data (200 synthetic tickets) ----------
random.seed(42)

PARTS = {
 "Access": (["I cannot","Unable to","Not able to","Struggling to"],
            ["login to my account","reset my password","access the shared drive","sign in to the portal","unlock my account","get my OTP code"],
            ["since morning","after the update","please help","again today","on my laptop","after changing password"]),
 "Billing": (["I was charged twice for","Wrong amount shown on","Need a refund for","Payment failed for","Incorrect tax added to"],
            ["my invoice","the monthly subscription","my last order","the annual plan","my credit card bill","the renewal"],
            ["please fix","this month","urgently","money got deducted","kindly check","and no receipt received"]),
 "Bug": (["App crashes when","Error 500 appears when","Page not loading when","Button not working when","Screen freezes when"],
            ["I open the dashboard","I submit the form","I upload a file","I click save","I open reports","I search for a record"],
            ["on mobile","after the latest update","every time","on Chrome","in production","since yesterday"]),
 "Feature Request": (["Please add","Would love to have","Can you support","Suggest adding","It would help to include"],
            ["dark mode","export to Excel","scheduled reports","multi-language support","a calendar view","bulk upload"],
            ["in the app","in the next release","for our team","to save time","for better usability","as an option"]),
 "General Query": (["How do I","Where can I","What is the way to","Can you tell me how to","Who do I contact to"],
            ["change my profile email","find the user manual","download past reports","update my phone number","view support hours","check my plan details"],
            ["","please guide","thanks","","just curious","for my account"]),
}

DATA = []
for cat, (a, b, c) in PARTS.items():
    seen = set()
    while len(seen) < 40:
        seen.add(f"{random.choice(a)} {random.choice(b)} {random.choice(c)}".strip())
    DATA += [(t, cat) for t in seen]
random.shuffle(DATA)

# ---------- 2. Routing + priority config ----------
TEAM = {
    "Access": "IT Support",
    "Billing": "Finance Team",
    "Bug": "Engineering",
    "Feature Request": "Product Team",
    "General Query": "Customer Support",
}
URGENT_WORDS = {"urgent", "urgently", "asap", "critical", "down", "crash", "crashes",
                "charged twice", "deducted", "locked", "immediately"}
CONF_THRESHOLD = 0.45
MODEL_PATH = "ticket_model.joblib"


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train(df):
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["category"], test_size=0.25, random_state=42, stratify=df["category"]
    )
    model = build_pipeline().fit(X_tr, y_tr)
    print(classification_report(y_te, model.predict(X_te), zero_division=0))
    model.fit(df["text"], df["category"])
    joblib.dump(model, MODEL_PATH)
    return model


def get_priority(text):
    t = clean(text)
    return "High" if any(w in t for w in URGENT_WORDS) else "Normal"


def process_ticket(model, text):
    probs = model.predict_proba([text])[0]
    idx = probs.argmax()
    category, conf = model.classes_[idx], float(probs[idx])
    routed = TEAM[category] if conf >= CONF_THRESHOLD else "Manual Review"
    return {
        "ticket": text,
        "category": category,
        "confidence": round(conf, 2),
        "priority": get_priority(text),
        "assigned_to": routed,
    }


if __name__ == "__main__":
    df = pd.DataFrame(DATA, columns=["text", "category"])
    model = train(df)

    samples = [
        "URGENT: app crashes every time I log in",
        "I was charged twice, please fix asap",
        "Can you add a calendar view?",
        "Where do I find the invoice PDF?",
    ]
    result = pd.DataFrame([process_ticket(model, s) for s in samples])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 50)
    print(result.to_string(index=False))
