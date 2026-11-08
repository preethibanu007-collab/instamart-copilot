import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, re, random, time, os
import pandas as pd
import mock_llm

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED = ["billing", "delivery", "technical", "fraud"]

ZERO = ('You CLASSIFY customer support tickets into exactly one category: '
        'billing, delivery, technical, fraud. '
        'Reply with JSON only: {"category": "...", "reason": "..."}')
FEW = ZERO + ('\nEXAMPLES:\n'
              'Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}\n'
              'Ticket: The rider never delivered [ORDER_ID]. -> {"category": "delivery", "reason": "not received"}\n'
              'Ticket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app crash"}\n'
              'Ticket: Someone hacked my account and used my wallet, this is fraud. -> {"category": "fraud", "reason": "unauthorized use"}\n'
              'Ticket: There is an unauthorized transaction on my card. -> {"category": "fraud", "reason": "unauthorized transaction"}')

PATTERNS = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            (re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            (re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

def redact(text, counts):
    for pat, ph in PATTERNS:
        text, n = pat.subn(ph, text)
        counts[ph] += n
    return text

def call_with_retry(messages, tid, log):
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            print(f"retry {tid} retry_number={n} wait={wait:.4f}s")
            log.append(wait)
            time.sleep(wait)

def validate(content):
    try:
        d = json.loads(content)
        if isinstance(d, dict) and d.get("category") in ALLOWED:
            return "classified_json", d["category"]
        return "flagged_invalid_output", ""
    except Exception:
        words = re.findall(r"[a-z]+", content.lower())
        found = {w for w in words if w in ALLOWED}
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""

def run(prompt, budget=None, quiet=False):
    df = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    mock_llm.reset()
    counts = {"[EMAIL]": 0, "[PHONE]": 0, "[ORDER_ID]": 0}
    log, rows, total, sent = [], [], 0, []
    for _, t in df.iterrows():
        text = str(t["text"])
        row = {"ticket_id": t["ticket_id"], "customer_id": t["customer_id"],
               "true_category": t["true_category"], "category": "", "tokens_used": 0}
        if text.strip() == "":
            row["status"] = "skipped_empty"
        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        else:
            clean = redact(text, counts)
            for pat, _ in PATTERNS:
                assert not pat.search(clean)
            if budget is not None and total >= budget:
                row["status"] = "budget_exhausted"
            else:
                msgs = [{"role": "system", "content": prompt},
                        {"role": "user", "content": clean}]
                sent.append(clean)
                r = call_with_retry(msgs, t["ticket_id"], log)
                used = r.usage.total_tokens
                total += used
                row["tokens_used"] = used
                row["status"], row["category"] = validate(r.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows), counts, log, total

def report(name, res, total):
    ok = res[res["status"].isin(["classified_json", "classified_keyword"])]
    acc = (ok["category"] == ok["true_category"]).mean() * 100 if len(ok) else 0
    print(f"\n{name}: classified={len(ok)} accuracy={acc:.1f}% "
          f"total_tokens={total} tokens/ticket={total/max(len(ok),1):.1f}")
    print(res["status"].value_counts().to_dict())
    return ok

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    z, counts, log, zt = run(ZERO)
    print("redactions:", counts, "total:", sum(counts.values()))
    print("retries:", len(log), "| waits within 0.045-0.0575:",
          all(0.045 <= w <= 0.0575 for w in log))
    ok = report("ZERO-SHOT", z, zt)
    print(pd.crosstab(ok["true_category"], ok["category"]))

    f, _, _, ft = run(FEW)
    report("FEW-SHOT", f, ft)

    d, _, _, dt = run(ZERO, budget=6000)
    d.to_csv("outputs/triage_budget_demo.csv", index=False)
    print("\nbudget demo (6000): exhausted =", (d["status"] == "budget_exhausted").sum())

    final, _, _, _ = run(FEW, budget=100000)
    seg = pd.read_csv("outputs/customer_segments.csv")
    seg = seg.set_index("customer_id")
    def route(r):
        if r["status"] in ["skipped_empty", "flagged_long",
                           "flagged_invalid_output", "budget_exhausted"]:
            return "human_review", "NORMAL"
        if r["category"] == "fraud":
            return "security_team", "HIGH"
        prem = r["customer_id"] in seg.index and seg.loc[r["customer_id"], "is_premium"] == 1
        return ("premium_desk", "HIGH") if prem else ("support_queue", "NORMAL")
    rt = final.apply(route, axis=1, result_type="expand")
    final["route"], final["priority"] = rt[0], rt[1]
    final["segment_name"] = final["customer_id"].map(
        seg["segment_name"]).fillna("")
    final[["ticket_id", "customer_id", "status", "category", "segment_name",
           "priority", "route", "tokens_used"]].to_csv(
        "outputs/triage_results.csv", index=False)
    print("triage_results rows:", len(final))
    print(final["route"].value_counts().to_dict())