"""골든셋으로 답변 품질을 잰다 (검색 → Gemini 답변 → Gemini 채점).

python evaluation/answer_eval.py                              # 05_golden_set/golden_set.jsonl (20문항)
python evaluation/answer_eval.py 05_golden_set/xxx.jsonl
python evaluation/answer_eval.py --neighbors=0                # 이웃 청크 없이 상위 5개만 (비교용)
python evaluation/answer_eval.py --answer-effort=default        # 답변의 생각 수준 (기본 low, E10 이전 방식은 default)
python evaluation/answer_eval.py --mode=agent --small=gemini-3.5-flash-lite  # Agent의 분류·근거 판단만 가벼운 모델로 (E9)
python evaluation/answer_eval.py --mode=agent                 # 기본 RAG 대신 Agent(agent/graph.py)로 답변 → agent_rows.jsonl
python evaluation/answer_eval.py --cases=POS006,NEG033            # 지정한 문항만
python evaluation/answer_eval.py --prompt=base                # 프롬프트 규칙 묶음 (base / A / AB / ABC, 기본 ABC, agent/prompts.py)
python evaluation/answer_eval.py 05_golden_set/golden_set_provided.jsonl --sample=30
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

import retrieval_eval as ret  # noqa: E402  검색 평가의 정답 만들기·점수를 그대로 씀
from common import EMBEDDING_MODEL_NAME, get_embeddings, log_cost  # noqa: E402
from graph import ANSWER_EFFORT, CLASSIFY_EFFORT, GRADE_EFFORT, MAX_RETRIES, build_agent  # noqa: E402
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


TEMPORARY = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "high demand", "Deadline", "timed out")


def with_retry(fn, what):
    # Gemini가 붐빌 때(503 high demand 등) 기다렸다 다시. 180문항을 돌리다 중간에 멈춘 적이 있어서 추가
    for wait in (30, 60, 120, 240, None):
        try:
            return fn()
        except Exception as e:
            if wait is None or not any(s in str(e) for s in TEMPORARY):
                raise
            print(f"  {what}: {str(e)[:80]} → {wait}초 뒤 다시")
            time.sleep(wait)


def evaluate(golden_path, embeddings, llm, judge_llm, neighbors, prompt, sample, mode, seed, small, answer_effort, cases=None):
    golden = [json.loads(line) for line in golden_path.open(encoding="utf-8") if line.strip()]
    # 기준 답안이 문서 밖의 내용(상식·계산)을 요구해서 평가에서 빼는 문항 (이유는 파일 안에)
    excluded = json.loads((ret.GOLDEN_DIR / "excluded_cases.json").read_text(encoding="utf-8"))
    golden = [g for g in golden if g["case_id"] not in excluded]
    name = golden_path.stem
    if cases:  # 지정한 문항만 (예: 오답 문항으로 프롬프트 수정 효과 보기)
        golden = [g for g in golden if g["case_id"] in cases]
        name = f"{name}_cases{len(golden)}"
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
    agent = build_agent(tool, prompt, small, answer_effort) if mode == "agent" else None
    by_model = {}  # Agent 모델별 [호출, 입력, 출력] (비용 계산용)

    # 중간 저장: 답변이 끝난 문항은 바로 파일에 적어 두고, 중간에 멈추면 다시 실행할 때 건너뛴다.
    # 같은 설정(질문 세트·모드·프롬프트·이웃·모델·생각 수준·청크 파일)일 때만 이어서 한다
    out_dir = ret.OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_md5 = hashlib.md5((ret.CHUNK_DIR / f"{CONFIG}.jsonl").read_bytes()).hexdigest()[:8]  # 청크 파일 지문 (runs.jsonl에도 기록)
    sig = hashlib.md5(json.dumps([mode, prompt, neighbors, small, answer_effort, chunks_md5]).encode()).hexdigest()[:8]
    partial = out_dir / f"_partial_{mode}_{sig}.jsonl"
    done = {}
    if partial.exists():
        for r in map(json.loads, partial.open(encoding="utf-8")):
            r["target"] = {tuple(t) for t in r["target"]} if r["target"] else None
            done[r["case_id"]] = r
        print(f"  중간 저장에서 이어서: {len(done)}문항은 건너뜀")

    # 1) 검색 + 답변 (문항마다 시간을 재려고 하나씩)
    rows, usage = [], {"answer": [0, 0], "judge": [0, 0]}
    for i, g in enumerate(golden, 1):
        if g["case_id"] in done:
            r = done[g["case_id"]]
            rows.append(r)
            usage["answer"][0] += r["input_tokens"]
            usage["answer"][1] += r["output_tokens"]
            for m, v in (r.pop("by_model", None) or {}).items():
                by_model[m] = [a + b for a, b in zip(by_model.get(m, [0, 0, 0]), v)]
            continue
        groups = BASIC_GROUPS if is_basic(g["case_id"]) else None
        target = targets.get(g["case_id"])
        start = time.perf_counter()
        if agent:
            out = with_retry(lambda: agent.invoke({"question": g["question"], "groups": groups}), g["case_id"])
            total_ms = (time.perf_counter() - start) * 1000
            context_docs, answer = out["docs"], out["answer"]
            calls, searches, tok = out["llm_calls"], out["searches"], (out["input_tokens"], out["output_tokens"])
            for m, v in out["by_model"].items():
                by_model[m] = [a + b for a, b in zip(by_model.get(m, [0, 0, 0]), v)]
            retrieval_ms = answer_ms = None
            extra_info = {"route": out["route"], "trace": out["trace"], "queries": out.get("queries"),
                          "retries": out.get("retries", 0)}
        else:
            context_docs = tool.search(g["question"], neighbors=neighbors, groups=groups)
            retrieval_ms = (time.perf_counter() - start) * 1000
            msg = with_retry(lambda: llm.invoke(ANSWER_PROMPT.format(extra=extra, context=to_context(context_docs),
                                                                     question=g["question"])), g["case_id"])
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
        with partial.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**rows[-1], "target": sorted(target) if target else None,
                                **({"by_model": out["by_model"]} if agent else {})}, ensure_ascii=False) + "\n")

    # 2) 채점 (4개씩 동시에). 붐벼서 실패한 문항은 하나씩 다시
    print("채점 중...")
    prompts = [JUDGE_PROMPT.format(question=g["question"], context=r["context"], reference=g["target_answer"],
                                   behavior=g["expected_behavior"], must_include=g["must_include"],
                                   must_not_include=g["must_not_include"], answer=r["answer"])
               for g, r in zip(golden, rows)]
    judged = judge_llm.batch(prompts, config={"max_concurrency": 4}, return_exceptions=True)
    judged = [with_retry(lambda p=p: judge_llm.invoke(p), f"채점 {r['case_id']}") if isinstance(j, Exception) else j
              for p, r, j in zip(prompts, rows, judged)]

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

    # 3) 저장 (다 끝났으니 중간 저장 파일은 지운다)
    prefix = "agent" if agent else "answer"
    with (out_dir / f"{prefix}_rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    partial.unlink(missing_ok=True)

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
    if agent:  # 노드마다 모델이 다를 수 있어서 모델별로 기록
        for m, (calls, i, o) in by_model.items():
            cost += log_cost(f"답변 평가 Agent 실행 ({name}, 프롬프트 {prompt})", m, calls, i, o, note=run_id)
    else:
        cost += log_cost(f"답변 평가 답변 생성 ({name}, 프롬프트 {prompt})", LLM_MODEL, len(rows), *usage["answer"], note=run_id)
    cost += log_cost(f"답변 평가 답변 채점 ({name}, 프롬프트 {prompt})", LLM_MODEL, len(rows), *usage["judge"], note=run_id)

    with (ret.OUT_DIR / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_id": run_id, "eval": "answer", "mode": mode,
            "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "golden_set": name, "count": len(rows),
            "params": {"config": CONFIG, "chunks_md5": chunks_md5, "search": "Hybrid", "k": K, "fetch_k": FETCH_K,
                       "neighbors": neighbors, "prompt": prompt,
                       "hybrid_weights": WEIGHTS, "embedding": EMBEDDING_MODEL_NAME,
                       "llm": LLM_MODEL, "judge": LLM_MODEL,
                       **({"answer_effort": answer_effort or ANSWER_EFFORT} if not agent else {}),
                       **({"max_retries": MAX_RETRIES, "classify_effort": CLASSIFY_EFFORT, "grade_effort": GRADE_EFFORT,
                           "small_model": small or LLM_MODEL, "answer_effort": answer_effort or ANSWER_EFFORT}
                          if agent else {})},
            "results": results,
            "tokens": usage, "cost_usd": round(cost, 4),
        }, ensure_ascii=False) + "\n")


def main():
    neighbors = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--neighbors=")), 1)
    prompt = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--prompt=")), "ABC")
    mode = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--mode=")), "rag")
    small = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--small=")), None)
    answer_effort = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--answer-effort=")), None)
    sample = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--sample=")), 0)
    seed = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--seed=")), 0)
    cases = next((set(a.split("=")[1].split(",")) for a in sys.argv[1:] if a.startswith("--cases=")), None)
    paths = [Path(p) for p in sys.argv[1:] if not p.startswith("--")] or [ret.GOLDEN_DIR / "golden_set.jsonl"]
    embeddings = ret.QueryCache(get_embeddings())
    key = os.environ["GEMINI_API_KEY"]
    # 답변 LLM의 생각 수준은 --answer-effort (E10-b). 채점 LLM은 항상 모델 기본값 (자를 바꾸지 않기)
    # 기본은 Agent와 같은 ANSWER_EFFORT(low), "default"면 모델 기본값
    rag_effort = (answer_effort or ANSWER_EFFORT) if mode == "rag" else None
    rag_effort = None if rag_effort == "default" else rag_effort
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=key, timeout=120, max_retries=2,
                                 **({"reasoning_effort": rag_effort} if rag_effort else {}))
    judge_llm = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=key, timeout=120,
                                       max_retries=2).with_structured_output(Judge, include_raw=True)
    for path in paths:
        evaluate(path, embeddings, llm, judge_llm, neighbors, prompt, sample, mode, seed, small, answer_effort, cases)


if __name__ == "__main__":
    main()
