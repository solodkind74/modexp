"""Generate synthetic bank/card notification messages with ground-truth JSON labels.

Writes data/{train,valid,test}.jsonl in the chat format mlx-lm expects.
The test split uses merchants never seen in training, so it measures generalization
(e.g. can the model guess that an unseen "Wingstop" is food?).
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SYSTEM = (
    "Extract the transaction from the bank/card message as JSON with keys: "
    "is_transaction (bool), type (\"debit\" or \"credit\"), amount (number), "
    "currency (ISO code), merchant (string), date (YYYY-MM-DD), "
    "card_last4 (string or null), category (one of: groceries, dining, transport, "
    "shopping, entertainment, bills, travel, health, income, transfer, other). "
    "If the year is missing, assume 2026. If the message is not a completed "
    "transaction, output {\"is_transaction\": false}. Output only JSON."
)

MERCHANTS = {
    "groceries": ["Whole Foods", "Trader Joe's", "Kroger", "Aldi", "Tesco", "Lidl", "Safeway",
                  "Carrefour", "Sainsbury's", "Publix", "Costco", "Rewe", "Edeka", "Wegmans"],
    "dining": ["Starbucks", "McDonald's", "Chipotle", "Pret A Manger", "Domino's", "Subway",
               "Uber Eats", "DoorDash", "Shake Shack", "KFC", "Wingstop", "Five Guys",
               "Dunkin'", "Nando's", "Deliveroo", "Taco Bell"],
    "transport": ["Uber", "Lyft", "Shell", "BP", "Chevron", "MTA NYC Transit", "TfL",
                  "Bolt", "Exxon", "Deutsche Bahn", "ParkMobile", "Citi Bike"],
    "shopping": ["Amazon", "IKEA", "Zara", "H&M", "Best Buy", "Target", "Walmart", "eBay",
                 "Apple Store", "Uniqlo", "Etsy", "Nike", "Decathlon", "Home Depot"],
    "entertainment": ["Netflix", "Spotify", "Steam", "PlayStation Store", "AMC Theatres",
                      "Disney+", "Ticketmaster", "YouTube Premium", "Nintendo eShop", "Audible"],
    "bills": ["Comcast", "Verizon", "AT&T", "PG&E", "Vodafone", "T-Mobile", "British Gas",
              "Con Edison", "Google One", "iCloud Storage"],
    "travel": ["Airbnb", "Booking.com", "Delta Air Lines", "Ryanair", "Marriott", "Expedia",
               "easyJet", "Hilton", "United Airlines", "Lufthansa"],
    "health": ["CVS Pharmacy", "Walgreens", "Boots", "Planet Fitness", "One Medical",
               "Rite Aid", "PureGym", "Equinox"],
}
INCOME = ["ACME Corp Payroll", "Stripe Payout", "Upwork", "IRS Treas Refund", "Globex Ltd Salary",
          "Initech Payroll", "PayPal Payout", "Fiverr"]
PEOPLE = ["John Smith", "Maria Garcia", "A. Petrov", "Li Wei", "Emma Brown", "K. Nakamura",
          "Sofia Rossi", "David Cohen", "Olga Ivanova", "James Lee"]

BANKS = ["Chase", "BofA", "Revolut", "Monzo", "Wise", "N26", "Citi", "Capital One", "HSBC", "Amex"]
CURRENCIES = {"USD": "$", "EUR": "€", "GBP": "£"}


def fmt_amount(rng, amount, cur):
    sym = CURRENCIES[cur]
    plain = f"{amount:,.2f}"
    euro = plain.replace(",", "X").replace(".", ",").replace("X", ".")
    style = rng.randint(0, 4)
    if style == 0:
        return f"{sym}{plain}"
    if style == 1:
        return f"{cur} {plain}"
    if style == 2:
        return f"{plain} {cur}"
    if style == 3 and cur == "EUR":
        return f"{euro} €"
    return f"{sym}{amount:.2f}"


def fmt_date(rng, d):
    style = rng.randint(0, 5)
    if style == 0:
        return d.strftime("%Y-%m-%d")
    if style == 1:
        return d.strftime("%b %-d")  # no year -> 2026
    if style == 2:
        return d.strftime("%-d %b %Y")
    if style == 3:
        return d.strftime("%d.%m.%Y")
    if style == 4:
        return d.strftime("%b %-d, %Y at %-I:%M %p")
    return d.strftime("%B %-d")  # no year -> 2026


def rand_amount(rng, category):
    lo, hi = {
        "groceries": (3, 250), "dining": (2, 90), "transport": (1.5, 120),
        "shopping": (5, 1500), "entertainment": (1, 80), "bills": (10, 300),
        "travel": (30, 2500), "health": (5, 200), "income": (50, 9000), "transfer": (5, 2000),
    }[category]
    return round(rng.uniform(lo, hi), 2)


def make_transaction(rng, merchants, income, people):
    d = datetime(2026, 1, 1) + timedelta(days=rng.randint(0, 364), minutes=rng.randint(0, 1439))
    cur = rng.choices(list(CURRENCIES), weights=[6, 3, 2])[0]
    card = f"{rng.randint(0, 9999):04d}"
    bank = rng.choice(BANKS)
    kind = rng.choices(["spend", "income", "transfer_in", "transfer_out"], weights=[14, 2, 1, 1])[0]

    if kind == "spend":
        category = rng.choice(list(merchants))
        merchant = rng.choice(merchants[category])
        typ = "debit"
    elif kind == "income":
        category, merchant, typ = "income", rng.choice(income), "credit"
    else:
        category, merchant = "transfer", rng.choice(people)
        typ = "credit" if kind == "transfer_in" else "debit"

    amount = rand_amount(rng, category)
    a, ds = fmt_amount(rng, amount, cur), fmt_date(rng, d)
    has_card = kind == "spend" and rng.random() < 0.75
    m_raw = merchant.upper() if rng.random() < 0.3 else merchant
    if rng.random() < 0.2 and kind == "spend":
        m_raw += f" #{rng.randint(100, 9999)}"  # label keeps the clean name

    if kind == "spend":
        tpls = [
            f"{bank}: Your card ending <CARD> was charged {a} at {m_raw} on {ds}.",
            f"Purchase of {a} at {m_raw} with card *<CARD>, {ds}. Not you? Call us.",
            f"[{bank}] {ds} {m_raw} -{a} card x<CARD>",
            f"You spent {a} at {m_raw} 💳 {ds}",
            f"{bank} Alert: A {a} transaction with {m_raw} was approved on your card ...<CARD> ({ds}).",
            f"Paid {a} to {m_raw} on {ds}. Avail bal: {fmt_amount(rng, rand_amount(rng, 'shopping') * 3, cur)}",
            f"POS DEBIT {ds} {m_raw} {a} CARD <CARD>",
        ]
        if not has_card:
            tpls = [t for t in tpls if "<CARD>" not in t]
    elif kind == "income":
        tpls = [
            f"{bank}: Deposit of {a} from {m_raw} received {ds}.",
            f"You received {a} from {m_raw} 💰 ({ds})",
            f"ACH CREDIT {ds} {m_raw} +{a}",
        ]
    elif kind == "transfer_in":
        tpls = [
            f"{m_raw} sent you {a} on {ds}.",
            f"{bank}: Incoming transfer {a} from {m_raw}, {ds}",
        ]
    else:
        tpls = [
            f"You sent {a} to {m_raw} on {ds}.",
            f"{bank}: Transfer of {a} to {m_raw} completed {ds}.",
        ]

    text = rng.choice(tpls)
    card_shown = "<CARD>" in text
    text = text.replace("<CARD>", card)
    label = {
        "is_transaction": True, "type": typ, "amount": amount, "currency": cur,
        "merchant": merchant, "date": d.strftime("%Y-%m-%d"),
        "card_last4": card if card_shown else None, "category": category,
    }
    return text, label


def make_non_transaction(rng, merchants):
    bank = rng.choice(BANKS)
    m = rng.choice(rng.choice(list(merchants.values())))
    code = rng.randint(100000, 999999)
    a = fmt_amount(rng, round(rng.uniform(5, 500), 2), rng.choice(list(CURRENCIES)))
    return rng.choice([
        f"{bank}: Your verification code is {code}. Do not share it with anyone.",
        f"Your {a} payment to {m} is scheduled for tomorrow.",
        f"{bank}: Your transaction of {a} at {m} was DECLINED. Reply Y if this was you.",
        f"Get 5% cashback at {m} this weekend with your {bank} card!",
        f"{bank}: Your statement is ready. Minimum payment due: {a}.",
        f"Reminder: {m} subscription renews in 3 days for {a}.",
        f"{bank}: New login to your account from iPhone. Wasn't you? Tap to secure.",
    ]), {"is_transaction": False}


def split_merchants(rng):
    """Hold out ~20% of merchants (and some income sources/people) for the test split."""
    train, test = {}, {}
    for cat, names in MERCHANTS.items():
        names = names[:]
        rng.shuffle(names)
        k = max(2, len(names) // 5)
        test[cat], train[cat] = names[:k], names[k:]
    return train, test


def build(rng, n, merchants, income, people):
    rows = []
    for _ in range(n):
        if rng.random() < 0.12:
            text, label = make_non_transaction(rng, merchants)
        else:
            text, label = make_transaction(rng, merchants, income, people)
        rows.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
        ]})
    return rows


def main():
    rng = random.Random(42)
    train_m, test_m = split_merchants(rng)
    out = Path("data")
    out.mkdir(exist_ok=True)
    splits = {
        "train": build(rng, 2000, train_m, INCOME[:6], PEOPLE[:7]),
        "valid": build(rng, 150, train_m, INCOME[:6], PEOPLE[:7]),
        "test": build(rng, 200, test_m, INCOME[6:], PEOPLE[7:]),
    }
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} examples")


if __name__ == "__main__":
    main()
