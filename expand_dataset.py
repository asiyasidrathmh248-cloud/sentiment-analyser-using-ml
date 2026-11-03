"""Expands dataset_original.csv with template-generated sentences -> dataset.csv.
The extra rows are synthetic (teaching data), see README."""
import itertools, random
import pandas as pd

random.seed(42)
subjects = ["the product", "the service", "this app", "the food", "the movie", "the hotel",
            "the delivery", "the support team", "this laptop", "the course", "the staff", "the phone"]
pos = ["great", "excellent", "amazing", "fantastic", "wonderful", "really good", "impressive", "superb", "delightful"]
neg = ["terrible", "awful", "horrible", "really bad", "disappointing", "poor", "frustrating", "useless", "annoying"]
pos_t = ["I love {s}, it is {a}.", "{S} was {a} and I am happy.", "Honestly {s} is {a}, highly recommended.",
         "I really enjoyed {s}, it was {a}.", "{S} exceeded my expectations, truly {a}.", "What {a} work by {s}, thank you!"]
neg_t = ["I hate {s}, it is {a}.", "{S} was {a} and I am upset.", "Honestly {s} is {a}, not recommended.",
         "I really disliked {s}, it was {a}.", "{S} failed my expectations, truly {a}.", "What {a} work by {s}, never again.",
         "{S} was not good at all.", "I am not happy with {s}."]
neu_t = ["{S} is available in two sizes.", "I read about {s} yesterday.", "{S} was mentioned in the meeting.",
         "The manual for {s} has twelve pages.", "I will check {s} on Monday.", "{S} arrived at noon.",
         "There is a review of {s} on the website.", "{S} was launched last year.", "I asked a question about {s}.",
         "{S} is located on the second floor."]

def gen(templates, adjs, n):
    combos = list(itertools.product(templates, subjects, adjs))
    random.shuffle(combos)
    out = []
    for t, s, a in combos[:n]:
        out.append(t.format(s=s, S=s[0].upper() + s[1:], a=a))
    return out

rows = []
rows += [(t, "positive") for t in gen(pos_t, pos, 80)]
rows += [(t, "negative") for t in gen(neg_t, neg, 80)]
rows += [(t, "neutral") for t in gen(neu_t, [""], 80)]
orig = pd.read_csv("dataset_original.csv")
df = pd.concat([orig, pd.DataFrame(rows, columns=["text", "sentiment"])]).drop_duplicates("text")
df.sample(frac=1, random_state=42).to_csv("dataset.csv", index=False)
print(df.sentiment.value_counts())
