# SKILL Benchmark Runner

该目录提供了一个与 `notebooks/Benchmarking.ipynb` 相同思路的自动化脚本，用来批量遍历 JSON 任务集、调用代理与工具（包含 `GeoSpatialProcessingSkill` 包装函数）并保存原始 JSON 结果以及 HTML 对话记录。

## 主要文件

- `run_skill_benchmark.py`：命令行脚本，加载任务集，调用 `geobenchx.generate_solutions` 运行批处理，可指定模型、温度、步数等参数。
- `results/`：脚本运行后自动生成的输出目录，其中包含：
  - `skill_skill_generated_solutions.json`（默认文件名，可通过 `--output-filename` 修改），记录所有任务的生成步骤；
  - `YYYY-mm-dd_HH-MM-SS_<model>_temp<temperature>/` 子目录，存放每个任务的 HTML 对话与工具调用轨迹，方便像原始评测那样快速浏览。

## 使用方法

```bash
python skill_test/skill_benchmark/run_skill_benchmark.py \
  --tasks-file skill_test/dataset/skill_tasks_and_reference_solutions.json \
  --output-filename skill_skill_generated_solutions.json \
  --results-folder skill_test/skill_benchmark/results \
  --model gpt-4o-2024-08-06 \
  --temperature 0.0
```

参数说明：

- `--tasks-file`：待处理任务集（JSON）；默认指向 `skill_test/dataset/skill_tasks_and_reference_solutions.json`。
- `--output-filename`：结果 JSON 名称（保存在 `--results-folder` 目录下）。
- `--results-folder`：输出目录，所有 JSON、HTML 会放在此处。
- `--model`、`--temperature`、`--max-steps` 等与 `Benchmarking.ipynb` 中一致。
- `--force`：若指定，则即便任务已有 `generated_solution` 也会重新尝试。
- `--no-history`：关闭 HTML 记录；默认开启以生成与原评测相同的可视化记录。

脚本会自动复用 `geobenchx/generate_solutions.py` 逻辑，因此要运行它需要与原项目相同的环境和 API Key（OpenAI/Anthropic/Gemini 等）配置。完成后即可在 `results` 下检视 JSON 结果与 HTML 对话，完全沿用主项目的评测与查看方式。
