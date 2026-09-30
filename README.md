# Auto Ticket Classification Flow

Flow: ticket text -> clean -> TF-IDF -> Logistic Regression -> category + confidence
-> priority rules -> route to team (or manual review if low confidence)

## Run
pip install -r requirements.txt
python ticket_classifier.py

## Customize
- Replace the synthetic DATA with your own CSV: df = pd.read_csv("tickets.csv") (columns: text, category)
- Edit TEAM, URGENT_WORDS, CONF_THRESHOLD for routing and priority rules
