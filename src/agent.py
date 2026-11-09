import sys, os, json, time, uuid, re, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import mock_llm
from rag import retrieve

DB = "data/instamart.db"
LEDGER = "outputs/refund_ledger.jsonl"
RUNS = "outputs/agent_runs.jsonl"
STATE = "state"

SYSTEM = ("You are a refund AGENT. Use the tools to look up the order, search the "
          "policies, calculate the refund, then issue or escalate it. "
          "Finish with one line: STATUS: explanation.")

class SimulatedCrash(Exception):
    pass

# ---------------- tools ----------------
def _orders_table(con):
    need = {"order_id", "status", "order_amount", "delivery_minutes"}
    for (t,) in con.execute("select name from sqlite_master where type='table'").fetchall():
        cols = {r[1] for r in con.execute(f"pragma table_info({t})")}
        if need <= cols:
            return t
    raise RuntimeError("orders table not found in data/instamart.db")

def get_order(order_id):
    con = sqlite3.connect(DB)
    try:
        t = _orders_table(con)
        row = con.execute(f"select order_id, status, order_amount, delivery_minutes "
                          f"from {t} where order_id = ?", (order_id,)).fetchone()
    finally:
        con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1],
            "order_amount": None if row[2] is None else float(row[2]),
            "delivery_minutes": None if row[3] is None else int(row[3])}

def search_policy(query):
    return {"results": retrieve(query, 3)}

def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and float(delivery_minutes) > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0.0}

def _ledger(tool, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "tool": tool, "order_id": order_id,
                            "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)

TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}

def _spec(name, desc, props):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props,
                           "required": list(props)}}

TOOLS_SPEC = [
    _spec("get_order", "Read one order (status, amount, delivery minutes) from the database.",
          {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents and return the best chunks.",
          {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund and write it to the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Escalate a refund above INR 1000 to a human lead.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]

# ---------------- helpers ----------------
def _chat(messages):
    for n in range(4):
        try:
            try:
                return mock_llm.chat(messages, tools=TOOLS_SPEC)
            except TypeError:
                return mock_llm.chat(messages, TOOLS_SPEC)
        except mock_llm.TransientLLMError:
            if n == 3:
                raise
            time.sleep(0.05)

def _g(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)

def _checkpoint(st):
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, st["run_id"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(st, f)

def _result(st, status, answer):
    return {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": status,
            "final_answer": answer, "steps": st["steps"],
            "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"]}

def _fallback(st, status, why):
    done = ", ".join(st["tools_executed"]) or "none"
    return _result(st, status, f"HANDOFF to a human agent. Reason: {why}. "
                               f"Tools completed so far: {done}.")

def _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after=None):
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= time_limit_seconds:
            return _fallback(st, "FALLBACK_TIMEOUT", "time limit reached")
        if st["llm_calls"] >= max_iterations:
            return _fallback(st, "FALLBACK_MAX_ITERATIONS", "max iterations reached")
        r = _chat(st["messages"])
        st["llm_calls"] += 1
        st["tokens_used"] += r.usage.total_tokens
        if st["tokens_used"] > token_limit:
            return _fallback(st, "FALLBACK_TOKEN_LIMIT", "token limit exceeded")
        msg = r.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            text = msg.content or ""
            return _result(st, text.split(":")[0].strip(), text)
        tc = calls[0]
        tid, name, args = _g(tc, "id"), _g(tc, "name"), _g(tc, "arguments")
        if isinstance(args, str):
            args = json.loads(args)
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tid, "name": name, "arguments": args}]})
        obs = TOOL_MAP[name](**args) if name in TOOL_MAP else {"error": "unknown tool"}
        st["messages"].append({"role": "tool", "tool_call_id": tid, "name": name,
                               "content": json.dumps(obs)})
        st["steps"].append({"tool": name, "arguments": args, "observation": obs})
        st["tools_executed"].append(name)
        st["step"] += 1
        _checkpoint(st)
        if crash_after and len(st["tools_executed"]) == crash_after:
            raise SimulatedCrash(f"simulated crash after tool {crash_after}")

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after=None):
    st = {"run_id": run_id or f"{scenario_id}_{uuid.uuid4().hex[:6]}",
          "scenario_id": scenario_id,
          "messages": [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": user_message}],
          "tokens_used": 0, "step": 0, "tools_executed": [], "steps": [], "llm_calls": 0}
    return _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after)

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(os.path.join(STATE, run_id + ".json"), encoding="utf-8") as f:
        st = json.load(f)
    return _loop(st, max_iterations, token_limit, time_limit_seconds)

def log_run(res):
    os.makedirs("outputs", exist_ok=True)
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: res[k] for k in ("scenario_id", "status", "final_answer",
                                                "llm_calls", "tokens_used", "steps")}) + "\n")

def refund_of(res):
    for s in res["steps"]:
        if s["tool"] == "calculate_refund":
            return s["observation"].get("refund_inr")
    return None

def col(df, *keys):
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c

if __name__ == "__main__":
    for p in (RUNS, LEDGER):
        if os.path.exists(p):
            os.remove(p)
    df = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)
    print("columns:", list(df.columns))
    c_id, c_msg = col(df, "scenario"), col(df, "message", "request", "text")
    c_ref = col(df, "expected_refund")
    mock_llm.reset()
    print("\n| scenario | status | refund_inr | model calls | tokens |\n|---|---|---|---|---|")
    first_msg = None
    for i, row in df.iterrows():
        sid = row[c_id] if c_id else f"S{i+1}"
        res = run_agent(row[c_msg], sid)
        log_run(res)
        if sid == "S1" or i == 0:
            first_msg = first_msg or row[c_msg]
        rf = refund_of(res)
        exp = row[c_ref] if c_ref else ""
        ok = ""
        if exp not in ("", None) and rf is not None:
            ok = " OK" if abs(float(exp) - rf) <= 0.01 else f" MISMATCH(expected {exp})"
        print(f"| {sid} | {res['status']} | {'' if rf is None else rf}{ok} | "
              f"{res['llm_calls']} | {res['tokens_used']} |")
    print("\nDemo token_limit=200 ->",
          run_agent(first_msg, "S1_tokdemo", token_limit=200)["status"])
    print("Demo time_limit_seconds=0 ->",
          run_agent(first_msg, "S1_timedemo", time_limit_seconds=0)["status"])
    rid = "S1_crash_demo"
    try:
        run_agent(first_msg, "S1", run_id=rid, crash_after=2)
    except SimulatedCrash as e:
        print("\ncrashed:", e)
    res = resume_agent(rid)
    print("resumed ->", res["status"], "| refund:", refund_of(res))
    with open(os.path.join(STATE, rid + ".json"), encoding="utf-8") as f:
        print("tools executed across crash+resume:", json.load(f)["tools_executed"])