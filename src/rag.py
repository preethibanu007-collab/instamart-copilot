import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POL = "data/policies"

def chunk_docs(size=40, overlap=10):
    chunks = []
    for path in sorted(glob.glob(os.path.join(POL, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        start = 0
        while True:
            chunks.append({"source": os.path.basename(path),
                           "text": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
            start += size - overlap
    return chunks

class Index:
    def __init__(self, size=40, overlap=10):
        self.chunks = chunk_docs(size, overlap)
        self.vec = TfidfVectorizer()
        self.mat = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"],
                 "score": float(sims[i]),
                 "text": self.chunks[i]["text"]} for i in top]

INDEX = Index()

def retrieve(query, k=3):
    return INDEX.retrieve(query, k)

SYSTEM = ("You are a POLICY ASSISTANT. Answer only from the supplied context. "
          "If the answer is not in the context, say: I could not find the information.")

def answer(question, index=None):
    index = index or INDEX
    hits = index.retrieve(question, 3)
    ctx = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{ctx}\nQUESTION:\n{question}"
    for _ in range(4):
        try:
            r = mock_llm.chat([{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": user}])
            return r.choices[0].message.content, [h["source"] for h in hits]
        except mock_llm.TransientLLMError:
            pass
    return "I could not find the information.", [h["source"] for h in hits]

def evaluate(index, label):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    mock_llm.reset()
    doc_ok = fact_ok = fb_ok = n_ans = n_un = 0
    broken = []
    for _, r in ev.iterrows():
        text, srcs = answer(r["question"], index)
        if r["expected_doc"] == "NONE":
            n_un += 1
            fb_ok += text.startswith("I could not find")
        else:
            n_ans += 1
            d = r["expected_doc"] in srcs
            f = r["expected_fact"] in text
            doc_ok += d
            fact_ok += f
            if not (d and f):
                broken.append(r["question"])
    print(f"\n{label}: doc in top3 {doc_ok}/{n_ans} | fact in answer {fact_ok}/{n_ans} "
          f"| fallback {fb_ok}/{n_un}")
    for q in broken:
        print("  BREAKS:", q)

if __name__ == "__main__":
    from collections import Counter
    print("chunks:", len(INDEX.chunks), dict(Counter(c["source"] for c in INDEX.chunks)))
    print([h["source"] for h in retrieve("refund policy late delivery")])
    evaluate(INDEX, "overlap 10")
    evaluate(Index(40, 0), "overlap 0")