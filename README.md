# modexp — fine-tune a small LLM to extract transactions as JSON

Qwen3-0.6B + LoRA via MLX, trained locally on an M5 MacBook Air (16 GB). ~4 min train, $0.

```sh
uv run python gen_data.py                        # synthetic data -> data/{train,valid,test}.jsonl
uv run python eval.py                            # baseline (no training)
uv run mlx_lm.lora -c lora_config.yaml           # train LoRA -> adapters/
uv run python eval.py --adapter adapters         # evaluate fine-tuned model
uv run mlx_lm.generate --model mlx-community/Qwen3-0.6B-bf16 --adapter-path adapters \
  --system-prompt "$(uv run python -c 'import gen_data; print(gen_data.SYSTEM)')" \
  --chat-template-config '{"enable_thinking": false}' \
  --prompt "Chase: Your card ending 4421 was charged \$18.40 at Blue Bottle Coffee on Oct 3."
```

Test set uses merchants/payers never seen in training.

| | Base | + LoRA (600 iters) |
|---|---|---|
| Exact match | 11.0% | 72.0% |
| type | 25.7% | 100% |
| amount | 78.9% | 83.4% |
| category | 11.4% | 86.3% |

## Ideas to try next
- **Amount digit stutter** (`1324.27` -> `13244.27`): try fewer iters (loss hit ~0 = overfitting), lower LR, or the 1.7B model.
- **Category for unseen merchants**: needs world knowledge -> bigger model (`mlx-community/Qwen3-1.7B-bf16`) or more merchant variety.
- Make data harder/realistic: typos, other languages, ambiguous formats, real SMS you have.
- Fuse + export: `mlx_lm.fuse --model ... --adapter-path adapters --export-gguf` -> run in Ollama/LM Studio.
