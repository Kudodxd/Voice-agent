"""Synthetic toy corpus (text only) for the mock smoke test: 5 domains x 6 talks; domain terms
recur across talks of the same domain, one talk-specific name never recurs."""
import _common  # noqa: F401
import argparse
import random
from pathlib import Path

from vagent.data.manifest import Utterance, write_manifest

TERMS = {
    "speech": ["音声認識", "音響モデル", "話者適応", "雑音抑圧", "言語モデル", "発話区間"],
    "medical": ["心筋梗塞", "動脈硬化", "臨床試験", "抗凝固薬", "血管内皮", "冠動脈"],
    "geology": ["堆積岩", "断層運動", "地殻変動", "花崗岩", "火山灰", "変成岩"],
    "economy": ["為替介入", "金融緩和", "需給ギャップ", "物価指数", "財政赤字", "国債利回り"],
    "lingui": ["敬語体系", "形態素", "方言分布", "音韻規則", "談話標識", "統語構造"],
}
NAMES = ["勅使河原", "早乙女", "東海林", "五十嵐", "長谷部", "御手洗", "大豆生田", "一番ヶ瀬", "九十九", "薬師寺",
         "宍戸", "毒島", "鬼頭", "百目鬼", "五月女", "雲母", "十文字", "二階堂", "不知火", "西園寺",
         "東雲", "霧島", "綾小路", "鷹司", "久我", "冷泉", "北大路", "錦織", "鷲尾", "鳳"]
GENERIC = ["天気", "問題", "時間", "会社", "学校", "仕事", "電車", "料理", "映画", "家族", "旅行", "結果", "方法",
           "説明", "研究", "発表", "評価", "性能", "概要", "課題", "実験", "比較", "データ", "今後", "研究者", "今日"]
EVAL_T = ["えー今日は{T}について発表します", "まず{T}の概要を説明します", "{T}の性能を評価しました",
          "次に{T}と{U}を比較しました", "この結果から{T}が重要だと分かります", "研究者の{P}は{T}を使っています",
          "えー{T}の課題は時間です", "最後に{T}の今後について話します", "実験では{T}のデータを使いました",
          "あのー{T}は難しい問題です", "{P}の方法では{U}を評価しました"]
TRAIN_T = ["今日は{N}について話します", "{N}の結果を説明します", "{N}と{M}を比較しました", "{N}は大切な問題です",
           "えー{N}の方法を評価しました", "{N}の今後について発表します", "研究者は{N}のデータを使いました",
           "{N}の概要と{M}の課題", "あのー{N}の性能は時間で決まります"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/toy")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    train = []
    for i in range(3000):
        s = rng.choice(TRAIN_T).format(N=rng.choice(GENERIC), M=rng.choice(GENERIC))
        train.append(Utterance(f"T{i:05d}", f"T{i // 20:04d}", s))
    write_manifest(train, out / "train.jsonl")
    utts, meta, names = [], ["talk_id\tspeaker\tdomain"], iter(NAMES)
    for d, (dom, terms) in enumerate(TERMS.items(), start=1):
        for k in range(6):
            tid = f"A{d:02d}M{k + 1:04d}"
            spk = f"{dom}_spk{k % 2}"
            meta.append(f"{tid}\t{spk}\t{dom}")
            talk_terms = rng.sample(terms, 3)
            name = next(names)
            for j in range(12):
                tmpl = rng.choice(EVAL_T)
                s = tmpl.format(T=rng.choice(talk_terms), U=rng.choice(talk_terms), P=name)
                utts.append(Utterance(f"{tid}_{j:04d}", tid, s, None, float(j), float(j + 1), spk))
    write_manifest(utts, out / "sessions.jsonl")
    (out / "talk_meta.tsv").write_text("\n".join(meta) + "\n", encoding="utf-8")
    (out / "static_hotwords.tsv").write_text("\n".join(TERMS["speech"][:3] + TERMS["medical"][:3]) + "\n", encoding="utf-8")
    print(f"train {len(train)} utts; sessions {len(utts)} utts / 30 talks -> {out}")


if __name__ == "__main__":
    main()
