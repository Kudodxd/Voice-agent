#!/usr/bin/env bash
# End-to-end smoke test with mock backends (no audio / GPU). Takes ~1 min on CPU.
set -euo pipefail
cd "$(dirname "$0")"
C=configs/toy.yaml
rm -rf work/toy runs_toy
python scripts/make_toy_data.py --out work/toy
python scripts/build_rare_vocab.py --config $C
python scripts/build_sequences.py --config $C --group-by domain --n-perm 3 --out work/toy/seq_domain.json
python scripts/build_sequences.py --config $C --group-by random --max-len 6 --n-perm 3 --out work/toy/seq_random.json
python scripts/go_nogo.py --config $C --sequences work/toy/seq_domain.json work/toy/seq_random.json --n-perm 3 \
  --out work/toy/go_nogo.csv
python scripts/run_longitudinal.py --config $C --exp main --sequences work/toy/seq_domain.json --n-perm 3 \
  --systems B0 B1 B2 B3 B4 B5 P P-noSupport P-noConf P-noRepeat P-noVerify P-surface P+static
python scripts/evaluate.py --config $C --exp main --baseline B0 --compare P --n-boot 500
python scripts/sweep_write_gate.py --config $C --exp main --system P-noRepeat
python scripts/make_poison.py --config $C --sequences work/toy/seq_domain.json --k 4 --out work/toy/poison.json
python scripts/run_longitudinal.py --config $C --exp poison --sequences work/toy/seq_domain.json --n-perm 3 \
  --poison work/toy/poison.json --systems B0 B3 P P-noVerify
python scripts/evaluate.py --config $C --exp poison --baseline B0 --compare P --n-boot 500
