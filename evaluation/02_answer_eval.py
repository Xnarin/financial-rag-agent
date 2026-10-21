"""골든셋으로 답변 품질을 잰다 (검색 → Gemini 답변 → Gemini 채점).

python evaluation/02_answer_eval.py                              # 05_golden_set/golden_set.jsonl (20문항)
python evaluation/02_answer_eval.py 05_golden_set/xxx.jsonl
python evaluation/02_answer_eval.py --neighbors=0                # 이웃 청크 없이 상위 5개만 (비교용)
python evaluation/02_answer_eval.py --mode=agent                 # 기본 RAG 대신 Agent(agent/graph.py)로 답변 → agent_rows.jsonl
python evaluation/02_answer_eval.py --prompt=base                # 프롬프트 규칙 묶음 (base / A / AB / ABC, 기본 ABC, agent/prompts.py)
python evaluation/02_answer_eval.py 05_golden_set/golden_set_provided.jsonl --sample=30
                                                                 # 무작위 30문항 (튜닝용 golden_set 문항은 빼고 뽑음, --seed=1로 다른 표본)

검색은 확정한 기준 설정: 300/50 + Hybrid(5:5), 상위 5개 + 이웃 청크를 답변 LLM에 넘김 (리랭킹 없음).
--neighbors=N: 검색된 청크마다 같은 문서의 앞뒤 N개 청크(쪽을 넘어가도 됨)를 같이 넘긴다 (기본 1, 0 = 끔).
  쪼개진 표의 다른 조각, 다음 쪽으로 이어지는 설명을 놓치지 않으려고.
기본 positive(POS001~060)·기본 negative(NEG) 문항은 과제 조건대로 용어집 + 표준약관만 검색한다.

채점 (문항마다 채점용 LLM 1회, RAGAS 대신 골든셋의 기준 답안·must_include·must_not_include를 그대로 씀)
- correctness: 기준 답안과 비교 (1 맞음 / 0.5 일부 / 0 틀림)
- faithfulness: 답의 주장 중 검색 근거로 뒷받침되는 비율 (RAGAS Faithfulness와 같은 방식, 지어낸 내용 확인)
- must_include: 꼭 들어가야 할 내용 충족 비율 / must_not: 피해야 할 오류를 하나라도 했으면 1
- behavior_ok: expected_behavior(근거 인용 답변, 거절, 추가 정보 요청 등)를 지켰나
- citation_ok: 답에 적은 [파일 p쪽] 중 정답 쪽이 있나 (규칙으로 계산, 정답 문서가 없는 문항은 제외)
- outcome: 성공 / 검색 실패(LLM에 넘긴 근거에 정답 쪽이 없음) / 답변 실패(근거는 넘겼는데 답이 틀림)

결과 → 07_eval/{골든셋 이름}/answer_rows.jsonl (문항별), answer_summary.txt, 07_eval/runs.jsonl (실행 이력),
       docs/비용기록.md (토큰·비용)
"""
import datetime
import hashlib
import importlib
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

ret = importlib.import_module("01_retrieval_eval")  # 검색 평가의 정답 만들기·점수를 그대로 씀
from common import EMBEDDING_MODEL_NAME, get_embeddings, log_cost  # noqa: E402
from graph import CLASSIFY_EFFORT, GRADE_EFFORT, MAX_RETRIES, build_agent  # noqa: E402
from prompts import ANSWER_PROMPT, extra_rules  # noqa: E402
from search import BASIC_GROUPS, CONFIG, FETCH_K, K, WEIGHTS, SearchTool, label, to_context  # noqa: E402

LLM_MODEL = "gemini-3.6-flash"

JUDGE_PROMPT = """RAG 답변을 채점하세요. 같은 뜻이면 표현이 달라도 맞은 것으로 봅니다.

[질문]
{question}

[검색 근거]
{context}

[기준 답안]
{reference}

[기대 행동] {behavior}
[꼭 들어가야 할 내용] {must_include}
[피해야 할 오류] {must_not_include}

[채점할 답변]
{answer}"""


class Claim(BaseModel):
    claim: str = Field(description="답변에 있는 사실 주장 하나 (인사말·출처 표기·확인처 안내 문장 제외)")
    supported: bool = Field(description="검색 근거에 이 내용이 있으면 true")


class Judge(BaseModel):
    claims: list[Claim]
    correctness: float = Field(description="기준 답안과 비교: 1 핵심이 모두 맞음, 0.5 일부만 맞거나 빠짐, 0 틀림")
    must_include: list[bool] = Field(description="[꼭 들어가야 할 내용] 항목마다 답변에 있으면 true (항목 수와 같은 길이)")
    must_not_include: list[bool] = Field(description="[피해야 할 오류] 항목마다 답변이 그 오류를 했으면 true (항목 수와 같은 길이)")
    behavior_ok: bool = Field(description="[기대 행동]을 지켰으면 true")
    reason: str = Field(description="판단 이유 한두 문장")


def is_basic(case_id):
    m = re.fullmatch(r"POS(\d+)", case_id)
    return case_id.startswith("NEG") or (m is not None and int(m.group(1)) <= 60)


def tokens(raw):
    # 출력 = 전체 - 입력 → 생각(thinking) 토큰까지 포함
    u = raw.usage_metadata
    return u["input_tokens"], u["total_tokens"] - u["input_tokens"]


def evaluate(golden_path, embeddings, llm, judge_llm, neighbors, prompt, sample, mode, seed):
    golden = [json.loads(line) for line in golden_path.open(encoding="utf-8") if line.strip()]
    name = golden_path.stem
    if sample:
        # 확인용 표본: 프롬프트를 고칠 때 본 golden_set 문항은 빼고, 항상 같은 문항이 뽑히게 seed 고정
        tuning = {json.loads(line)["case_id"] for line in (ret.GOLDEN_DIR / "golden_set.jsonl").open(encoding="utf-8")}
        pool = [g for g in golden if g["case_id"] not in tuning]
        for earlier in range(seed):  # seed N 표본은 앞 seed 표본과 겹치지 않게 (앞 표본을 보고 고친 뒤 새 문항으로 확인하려고)
            used = {g["case_id"] for g in random.Random(earlier).sample(pool, sample)}
            pool = [g for g in pool if g["case_id"] not in used]
        golden = random.Random(seed).sample(pool, sample)
        name = f"{name}_sample{sample}" + (f"_seed{seed}" if seed else "")
    extra = extra_rules(prompt)
    targets = ret.build_targets(golden)
    run_id = f"{mode}-{name}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    print(f"\n######## {golden_path.name}: {len(golden)}문항, run_id={run_id}")

    new_questions = {g["question"] for g in golden} - embeddings.cache.keys()
    for g in golden:
        embeddings.embed_query(g["question"])
    tool = SearchTool(embeddings)
    agent = build_agent(tool, prompt) if mode == "agent" else None

    # 1) 검색 + 답변 (문항마다 시간을 재려고 하나씩)
    rows, usage = [], {"answer": [0, 0], "judge": [0, 0]}
    for i, g in enumerate(golden, 1):
        groups = BASIC_GROUPS if is_basic(g["case_id"]) else None
        target = targets.get(g["case_id"])
        start = time.perf_counter()
        if agent:
            out = agent.invoke({"question": g["question"], "groups": groups})
            total_ms = (time.perf_counter() - start) * 1000
            context_docs, answer = out["docs"], out["answer"]
            calls, searches, tok = out["llm_calls"], out["searches"], (out["input_tokens"], out["output_tokens"])
            retrieval_ms = answer_ms = None
            extra_info = {"route": out["route"], "trace": out["trace"], "queries": out.get("queries"),
                          "retries": out.get("retries", 0)}
        else:
            context_docs = tool.search(g["question"], neighbors=neighbors, groups=groups)
            retrieval_ms = (time.perf_counter() - start) * 1000
            msg = llm.invoke(ANSWER_PROMPT.format(extra=extra, context=to_context(context_docs), question=g["question"]))
            total_ms = (time.perf_counter() - start) * 1000
            answer_ms = total_ms - retrieval_ms
            answer, calls, searches, tok = msg.text, 1, 1, tokens(msg)
            extra_info = {}
        for k, v in zip((0, 1), tok):
            usage["answer"][k] += v
        # 검색 순위가 붙은 청크 (기본 RAG는 상위 5개, Agent는 검색마다 상위 5개를 모은 것)
        docs = sorted((d for d in context_docs if d.metadata["rank"]), key=lambda d: d.metadata["rank"])
        context = to_context(context_docs)
        print(f"  [{i}/{len(golden)}] {g['case_id']} {total_ms / 1000:.1f}초, LLM {calls}회, 검색 {searches}회"
              + (f", {' → '.join(extra_info['trace'])}" if agent else ""))
        rows.append({"run_id": run_id, "case_id": g["case_id"], "set": g["set"], "category": g["category"],
                     "context_type": g["context_type"], "question": g["question"], "answer": answer,
                     "retrieved": [label(d) for d in docs], "chunk_ids": [d.id for d in context_docs],
                     "context_chunks": len(context_docs), "context": context,
                     "retrieval_ms": retrieval_ms, "answer_ms": answer_ms, "total_ms": total_ms, "target": target,
                     "llm_calls": calls, "searches": searches, "input_tokens": tok[0], "output_tokens": tok[1],
                     **extra_info,
                     "retrieval_hit": int(any(ret.is_relevant(d, target) for d in docs)) if target else None,  # 검색 순위 청크
                     "context_hit": int(any(ret.is_relevant(d, target) for d in context_docs)) if target else None})

    # 2) 채점 (4개씩 동시에)
    print("채점 중...")
    judged = judge_llm.batch([
        JUDGE_PROMPT.format(question=g["question"], context=r["context"], reference=g["target_answer"],
                            behavior=g["expected_behavior"], must_include=g["must_include"],
                            must_not_include=g["must_not_include"], answer=r["answer"])
        for g, r in zip(golden, rows)], config={"max_concurrency": 4})

    for r, j in zip(rows, judged):
        for k, v in zip((0, 1), tokens(j["raw"])):
            usage["judge"][k] += v
        p = j["parsed"]
        cited = set(re.findall(r"\[([^\[\]]+?\.pdf) p(\d+)\]", r["answer"]))
        target = r.pop("target")
        r.update({
            "correctness": p.correctness,
            # 근거 없이 답한 경우(Agent의 범위 밖 안내)는 근거 대비 비율을 잴 수 없어서 뺀다
            "faithfulness": None if not r["context_chunks"] else sum(c.supported for c in p.claims) / len(p.claims) if p.claims else 1.0,
            "must_include": statistics.mean(p.must_include) if p.must_include else 1.0,
            "must_not": int(any(p.must_not_include)),
            "behavior_ok": int(p.behavior_ok),
            "citation_ok": None if not target else int(any(f == cf and (pg is None or pg == int(cp))
                                                          for cf, cp in cited for f, pg in target)),
            "unsupported": [c.claim for c in p.claims if not c.supported],
            "judge_reason": p.reason,
        })
        passed = r["correctness"] >= 0.5 and not r["must_not"] and r["behavior_ok"]
        r["outcome"] = "성공" if passed else "검색 실패" if r["context_hit"] == 0 else "답변 실패"
        del r["context"]

    # 3) 저장
    out_dir = ret.OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "agent" if agent else "answer"
    with (out_dir / f"{prefix}_rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = ("correctness", "faithfulness", "must_include", "must_not", "behavior_ok", "citation_ok",
               "context_chunks", "llm_calls", "searches", "input_tokens", "output_tokens", "total_ms")

    def mean(rs, m):
        values = [r[m] for r in rs if r[m] is not None]
        return round(statistics.mean(values), 3) if values else None

    groups = {"전체": rows, **{s: [r for r in rows if r["set"] == s] for s in ("positive", "negative")}}
    results = {group: {"count": len(rs), **{m: mean(rs, m) for m in metrics},
                      **{o: sum(r["outcome"] == o for r in rs) for o in ("성공", "검색 실패", "답변 실패")}}
               for group, rs in groups.items() if rs}
    report = [f"##### {name} / {CONFIG} Hybrid / 이웃 청크 ±{neighbors} / 프롬프트 {prompt} / {"Agent" if agent else "기본 RAG"} / {LLM_MODEL} ({run_id})"]
    for group, res in results.items():
        report.append(f"\n=== {group} ===\n" + "\n".join(f"{k:>14}: {v}" for k, v in res.items()))
    report.append("\n=== 실패 문항 ===")
    report += [f"{r['case_id']} [{r['outcome']}] {r['judge_reason']}" for r in rows if r["outcome"] != "성공"]
    print("\n".join(report))
    (out_dir / f"{prefix}_summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    # 4) 비용 기록
    cost = 0.0
    if new_questions:
        cost += log_cost(f"답변 평가 질문 임베딩 ({name})", EMBEDDING_MODEL_NAME, len(new_questions),
                         round(sum(map(len, new_questions)) * 0.66), note=f"추정 (글자 × 0.66), {run_id}")
    for step, step_name in (("answer", "Agent 실행" if agent else "답변 생성"), ("judge", "답변 채점")):
        calls = sum(r["llm_calls"] for r in rows) if step == "answer" else len(rows)
        cost += log_cost(f"답변 평가 {step_name} ({name}, 프롬프트 {prompt})", LLM_MODEL, calls, *usage[step], note=run_id)

    # 청크 파일 지문: 청킹을 바꾸면 값이 달라져서 어떤 청크로 돌린 결과인지 구분된다
    chunks_md5 = hashlib.md5((ret.CHUNK_DIR / f"{CONFIG}.jsonl").read_bytes()).hexdigest()[:8]
    with (ret.OUT_DIR / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_id": run_id, "eval": "answer", "mode": mode,
            "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "golden_set": name, "count": len(rows),
            "params": {"config": CONFIG, "chunks_md5": chunks_md5, "search": "Hybrid", "k": K, "fetch_k": FETCH_K,
                       "neighbors": neighbors, "prompt": prompt,
                       "hybrid_weights": WEIGHTS, "embedding": EMBEDDING_MODEL_NAME,
                       "llm": LLM_MODEL, "judge": LLM_MODEL,
                       **({"max_retries": MAX_RETRIES, "classify_effort": CLASSIFY_EFFORT, "grade_effort": GRADE_EFFORT}
                          if agent else {})},
            "results": results,
            "tokens": usage, "cost_usd": round(cost, 4),
        }, ensure_ascii=False) + "\n")


def main():
    neighbors = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--neighbors=")), 1)
    prompt = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--prompt=")), "ABC")
    mode = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--mode=")), "rag")
    sample = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--sample=")), 0)
    seed = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--seed=")), 0)
    paths = [Path(p) for p in sys.argv[1:] if not p.startswith("--")] or [ret.GOLDEN_DIR / "golden_set.jsonl"]
    embeddings = ret.QueryCache(get_embeddings())
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=os.environ["GEMINI_API_KEY"], timeout=120, max_retries=2)
    judge_llm = llm.with_structured_output(Judge, include_raw=True)
    for path in paths:
        evaluate(path, embeddings, llm, judge_llm, neighbors, prompt, sample, mode, seed)


if __name__ == "__main__":
    main()
