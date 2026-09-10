# interleaved_rope (rope/interleaved_rope)

## Task Description

Merge three already-rotated temporal, height, and width RoPE streams by
selecting a source stream for each feature column. This layout is used by
multimodal models such as Qwen2-VL.

## Interface

```python
def interleaved_rope(x, mrope_section):
```

- `x` has shape `[3, S, D]` and contains temporal, height, and width streams.
- `mrope_section` is a three-element integer sequence whose sum is `D // 3`.
- The output has shape `[S, D]` and the same dtype and device as `x`.

## Definition

For column `d`, select `x[1, :, d]` when `d % 3 == 1` and
`d < 3 * mrope_section[1]`. Select `x[2, :, d]` when `d % 3 == 2` and
`d < 3 * mrope_section[2]`. Select `x[0, :, d]` otherwise.
