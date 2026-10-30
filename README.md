# csj-voice-agent

Codebase thí nghiệm cho đề tài **memory từ hiếm xuyên phiên, training-free** (voice agent) trên CSJ:
tự phát hiện từ hiếm không cần nhãn → ghi vào memory qua cổng xác minh âm học (kana-CTC) →
đọc lại ở các phiên sau có xác minh → đo cải thiện tích lũy và lan truyền lỗi.

```
src/vagent/
  agent.py                 vòng lặp mỗi phiên: CTC → read(verify) → decode → extract → gate → write
  memory/store.py          MemoryEntry + chính sách dung lượng/quên (unbounded|lru|decay|confidence),
                           prune theo contradiction
  memory/write_policy.py   trích ứng viên + cổng ghi (none|text_freq|glossary|acoustic|label)
  memory/read_policy.py    chọn hotword: verify bằng kana-CTC hoặc retrieval text-only; 3 kiểu biểu diễn
  matching/fuzzy.py        Levenshtein.ratio trên cửa sổ trượt len±1 (không cần ranh giới từ)
  text/                    chuẩn hóa, fugashi/UniDic, định nghĩa từ hiếm (ngoài top-N của CSJ train)
  asr/                     interface CTCBackend/Decoder, HF kana-CTC, Whisper prompt, mock, cache
  eval/                    CER / B-CER / U-CER, term P/R, harmful change, memory quality, bootstrap
scripts/                   prepare → vocab → sequences → go/no-go → run → evaluate → sweep / poison → export
configs/base.yaml          CSJ thật + định nghĩa toàn bộ system (B0–B5, P, ablation)
configs/toy.yaml           smoke test với mock backend
site/                      website trực quan hóa (index.html + data/results.js)
```

## Cài đặt & kiểm tra nhanh

```bash
pip install -e ".[asr,dev]"
pytest -q tests
./run_toy.sh          # ~10 s, CPU, không cần audio
```
`run_toy.sh` chạy toàn bộ pipeline trên corpus tổng hợp với backend mock. Con số của nó **chỉ để
kiểm tra code chạy đúng**, không mang ý nghĩa thực nghiệm.

## Quy trình trên CSJ

```bash
C=configs/base.yaml
# 0. Manifest từ data dir kiểu Kaldi/ESPnet (wav.scp, segments, text[, utt2spk])
python scripts/prepare_csj.py --data-dir $ESPNET/egs2/csj/asr1/data/train_nodup --out work/csj/manifests/train_nodup.jsonl
python scripts/prepare_csj.py --data-dir $ESPNET/egs2/csj/asr1/data/eval1 --out work/csj/manifests/eval1.jsonl
#    (gộp các talk dùng làm phiên vào work/csj/manifests/sessions.jsonl)
# 1. Định nghĩa từ hiếm từ CSJ train
python scripts/build_rare_vocab.py --config $C
# 2. Chuỗi phiên (+ nhóm random làm đối chứng)
python scripts/build_sequences.py --config $C --group-by domain --out work/csj/seq_domain.json
python scripts/build_sequences.py --config $C --group-by random --out work/csj/seq_random.json
# 3. Go/no-go: tỉ lệ tái xuất hiện từ hiếm giữa các phiên (chỉ cần reference)
python scripts/go_nogo.py --config $C --sequences work/csj/seq_domain.json work/csj/seq_random.json --out work/csj/go_nogo.csv
# 4. Chạy các system (resume được; CTC và decoder được cache theo utt + hotword)
python scripts/run_longitudinal.py --config $C --exp main --sequences work/csj/seq_domain.json \
    --n-perm 5 --systems B0 B1 B2 B3 B4 B5 P
# 5. Đánh giá
python scripts/evaluate.py --config $C --exp main --baseline B0 --compare P
```

## System và thí nghiệm

| ID | Cấu hình |
|---|---|
| B0 | không biasing |
| B1 | static list (`data.static_list`), lọc bằng CTC |
| B2 | glossary trong phiên, two-pass theo tần suất, không lưu qua phiên |
| B3 | memory text-only: ghi mọi ứng viên lặp ≥2 lần, đọc không xác minh, không quên |
| B4 | cổng ghi dùng reference của phiên (trần của write policy) |
| B5 | oracle list = từ hiếm trong reference của phiên hiện tại |
| **P** | cổng ghi âm học (support, conf, repeat) + đọc có xác minh + policy `confidence` |

- **Exp1 (write policy):** `P-noSupport`, `P-noConf`, `P-noRepeat`; đường precision–recall offline:
  `scripts/sweep_write_gate.py --exp main --system P-noRepeat` (mọi ứng viên, kể cả bị loại, đều được log
  trong `decisions.jsonl`). Khoảng cách P → B4 là con số trung tâm của RQ1.
- **Exp2 (longitudinal, lan truyền lỗi):** `by_step.csv`, `memory_by_step.csv`; `P-noVerify`;
  dung lượng `P-unbounded`, `P-lru`, `P-decay`; poisoning:
  ```bash
  python scripts/make_poison.py --config $C --sequences work/csj/seq_domain.json --k 5 --out work/csj/poison.json
  python scripts/run_longitudinal.py --config $C --exp poison --poison work/csj/poison.json \
      --sequences work/csj/seq_domain.json --n-perm 5 --systems B0 B3 P P-noVerify
  ```
- **Exp3 (biểu diễn):** `P-surface`, `P-mecabRead` (reading từ MeCab thay vì từ cửa sổ CTC), `P-context`.
- **Exp4 (static + dynamic):** `P+static` so với `B1` và `P`.

Thêm system mới: khai báo trong `systems:` (override lên `agent:`; có thể dùng `_extends_: P`).
Override nhanh từ CLI: `-o agent.read.top_k=50 backends.decoder.kwargs.num_beams=4`.

## Chỉ số (`runs/<exp>/eval/`)

- `CER`, `B-CER` (lỗi ký tự trong span từ hiếm của reference), `U-CER` (phần còn lại). Insertion được
  tính vào B chỉ khi cả hai ký tự lân cận đều thuộc span từ hiếm.
- `term_R` (trên các lần xuất hiện từ hiếm trong reference), `term_P` (trên từ hiếm ∪ hotword được đưa vào).
- `harmful`: tỉ lệ ký tự đúng ở baseline (B0) nhưng sai ở system. `induced_fp_per_1k`: số lần hotword được
  đưa vào xuất hiện thừa so với reference, trên 1000 ký tự.
- Memory sau mỗi phiên: `mem_precision` (entry có thực sự được nói ở các phiên ≤ t), `mem_poison`
  (số entry poison còn sống), `coverage_next` (tỉ lệ từ hiếm của phiên t+1 đã có trong memory).
- `bootstrap.csv`: paired bootstrap theo cụm phiên (seq, perm, step).

## Website trực quan hóa

Bản online: https://kudodxd.github.io/Voice-agent/ (GitHub Pages, tự deploy `site/` mỗi khi `site/` thay đổi trên `main`
qua `.github/workflows/pages.yml`; cần bật một lần: Settings → Pages → Source: **GitHub Actions**).

`site/index.html` là một trang tĩnh (không cần build, mở trực tiếp được cả từ `file://`) trình bày dự án:
sơ đồ vòng lặp agent (bấm vào từng khối để xem file và tham số), replay từng phiên (memory sau mỗi talk,
quyết định của cổng ghi, từng utterance B0 so với P), playground chỉnh ngưỡng cổng ghi trên các ứng viên đã log,
demo fuzzy match trên kana, bảng system, kết quả đánh giá và bản đồ code. Nút ở góc trên bên phải chuyển
ngôn ngữ (English / Tiếng Việt / 日本語); lựa chọn được lưu trong trình duyệt, lần đầu theo ngôn ngữ của trình duyệt.
Dữ liệu nằm trong
`site/data/results.js`, sinh từ thư mục `runs/` sau khi đã chạy `evaluate.py` (và `sweep_write_gate.py`):

```bash
python scripts/export_site_data.py --config configs/toy.yaml --exps main poison    # sau ./run_toy.sh
python scripts/export_site_data.py --config $C --exps main poison                  # số liệu CSJ thật
```
Bản commit sẵn dùng dữ liệu toy (mock backend), nên các con số chỉ để minh họa; trang tự ghi rõ điều này.
Replay mặc định là `P` so với `B0` (`--trace-system`, `--trace-baseline`); go/no-go được đọc từ
`<work_dir>/go_nogo.csv` (`go_nogo.py --out`).

## Gắn pipeline của bạn

Viết một class kế thừa `vagent.asr.base.Decoder` (mẫu: `src/vagent/asr/external_template.py`), nhận
`hotwords: list[Hotword(surface, reading, context)]` và `ctc: CTCResult`, rồi trỏ config tới nó:

```yaml
backends:
  decoder: {_replace_: true, type: external, class: mypkg.speechllm:MySpeechLLMDecoder,
            cache_tag: speechllm_v1, kwargs: {ckpt: /path}}
```
Tương tự cho CTC (`type: hf` dùng được với bất kỳ `AutoModelForCTC` có vocab kana; hoặc `type: external`).
Nếu decoder của bạn tự lọc hotword bằng CTC bên trong, hãy tắt `read.verify` ở các system tương ứng để
không lọc hai lần. Đổi `cache_tag` mỗi khi đổi model hoặc prompt, nếu không sẽ đọc nhầm cache cũ.

## Cần kiểm tra trước khi chạy thật

1. **Rò rỉ dữ liệu.** Nếu CTC head, encoder hoặc LLM đã thấy CSJ train, chỉ được dùng các talk ngoài
   tập đó (eval1–3 / held-out) làm phiên trong `sessions.jsonl`.
2. **Metadata chuỗi phiên.** `talk_meta.tsv` (`talk_id, speaker, domain`) phải tự dựng từ tài liệu CSJ.
   `--group-by prefix` chỉ là heuristic theo ký tự đầu của talk ID; cần đối chiếu ý nghĩa của ID với
   tài liệu CSJ trước khi dùng.
3. **Định dạng `text`.** `prepare_csj.py` bỏ hậu tố `+POS` nếu có và tạo `text_nofiller` bằng cách bỏ token
   `感動詞`. Kiểm tra file `text` của recipe bạn dùng; chọn một biến thể filler làm chính
   (`data.text_field`) và giữ cố định cho mọi system.
4. **Đơn vị từ.** Mặc định là UniDic (unidic-lite) với noun compound được gộp, không phải SUW của CSJ.
   Nên dùng UniDic đầy đủ (`tokenizer.mecab_args`) cho số liệu cuối cùng và ghi rõ trong bài.
5. **Giới hạn đã biết.** Đếm term dùng substring không chồng lấn (từ là chuỗi con của từ khác bị đếm
   cả hai). Sweep offline bỏ qua vòng phản hồi memory → decode, nên chỉ dùng để chọn điểm hoạt động.
   Poison đồng âm khớp đúng cách đọc nên xác minh âm học không chặn được nó; đây là trường hợp cần phân
   tích riêng ở Exp3.
