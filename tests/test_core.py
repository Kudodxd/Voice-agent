import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vagent.asr.base import CTCResult, Hotword  # noqa: E402
from vagent.config import deep_merge  # noqa: E402
from vagent.eval.metrics import utt_stats  # noqa: E402
from vagent.matching.fuzzy import best_match  # noqa: E402
from vagent.memory.read_policy import Reader  # noqa: E402
from vagent.memory.store import MemoryStore  # noqa: E402
from vagent.memory.write_policy import Candidate, WriteGate  # noqa: E402
from vagent.text.lexicon import RareTagger, Span, Tokenizer  # noqa: E402
from vagent.text.normalize import kana_key, normalize_text, strip_pos  # noqa: E402

TOK = Tokenizer()


def test_normalize():
    assert strip_pos("えー+感動詞 今日+名詞 は+助詞") == "えー今日は"
    assert strip_pos("えー+感動詞 今日+名詞", ["感動詞"]) == "今日"
    assert normalize_text("今日は、 ＡＢＣ。") == "今日はABC"
    assert kana_key("オンセー") == "おんせー"


def test_rare_compound():
    rt = RareTagger(TOK, common={"認識"})
    assert [s.surface for s in rt.rare_spans("音声認識の照会")] == ["音声認識", "照会"]


def test_fuzzy():
    assert best_match("しょうかい", "えーしょうかいです").score == 1.0
    m = best_match("おんせいにんしき", "えーおんせーにんしきの")
    assert 0.8 < m.score < 1.0 and m.start == 2
    assert best_match("あい", "あうい").score == 0.0  # short needle: exact only


def test_gate_and_memory():
    g = WriteGate(mode="acoustic", min_support=0.85, min_conf=0.6, min_repeat=2)
    cs = [Candidate("照会", "しょうかい", "しょうかい", "", f"u{i}", 1.0, 0.9) for i in range(2)]
    cs.append(Candidate("紹介", "しょうかい", "", "", "u0", 0.3, 0.9))
    acc = {d.surface: d.accepted for d in g.decide(cs)}
    assert acc == {"照会": True, "紹介": False}
    m = MemoryStore(policy="lru", max_size=1)
    m.upsert("A", "", [], 0, 1, 1.0, 1.0)
    m.upsert("B", "", [], 1, 1, 1.0, 1.0)
    m.end_session(1)
    assert [e.surface for e in m.values()] == ["B"]
    m2 = MemoryStore(prune_on_contradiction=True, contradiction_margin=2)
    m2.upsert("X", "", [], 0, 1, 1.0, 1.0)
    m2.feedback("X", False); m2.feedback("X", False); m2.end_session(0)
    assert len(m2) == 0


def test_reader_verification():
    m = MemoryStore()
    m.upsert("照会", "しょうかい", [], 0, 2, 1.0, 1.0)
    m.upsert("浜名湖", "はまなこ", [], 0, 2, 1.0, 1.0)
    r = Reader(TOK, verify=True, min_score=0.8)
    off = r.select(m.values(), [], CTCResult("えーしょうかいします", [1.0] * 10))
    assert [o.hotword.surface for o in off] == ["照会"]
    off = Reader(TOK, verify=False, verify_static=False).select(m.values(), [Hotword("音声")], None)
    assert {o.hotword.surface for o in off} == {"照会", "浜名湖", "音声"}


def test_metrics():
    ref, hyp = "今日は照会です", "今日は紹介です"
    st = utt_stats(ref, hyp, [Span("照会", "しょうかい", 3, 5)], {"照会"}, base_hyp=ref)
    assert (st.n_err, st.err_rare, st.err_u) == (2, 2, 0)
    assert (st.rec_tp, st.rec_fn, st.harmful, st.base_correct) == (0, 1, 2, 7)


def test_deep_merge_replace():
    base = {"a": {"x": 1, "y": 2}}
    assert deep_merge(base, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}}
    assert deep_merge(base, {"a": {"_replace_": True, "z": 0}}) == {"a": {"z": 0}}
