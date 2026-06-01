llama-cli \
  --model gemma-4-26B-A4B-it-Q8_0.gguf \
  --model-draft gemma-4-26B-A4B-it-assistant-GGUF/\
wikitext-2-raw_ik-llama-mtp_drafter-conservative/\
gemma-4-26B-A4B-it-assistant-Q8_0.gguf \
  --spec-type mtp --draft-max 3 --draft-p-min 0.0 --spec-autotune \
  -cnv --color --jinja --special \
  -sm graph -smgs -sas -mea 256 --split-mode-f32 \
  --temp 0.7 -t 8 --parallel 8 \
  --cpu-moe --merge-up-gate-experts \
  --flash-attn on --mla-use 3 \
  --mlock --run-time-repack --no-kv-offload
\\\
--spec-type mtp --draft-max 3 --draft-p-min 0.0 --spec-autotune
