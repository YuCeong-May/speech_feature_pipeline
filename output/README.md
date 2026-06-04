# Output Directory

本目录保存脚本运行后的生成产物，包括特征 CSV、字段名映射表、标准化 WAV、强制对齐结果和对齐后指标。

这些文件不是特征定义的权威来源。修改代码或切换版本后，请重新运行：

```bash
python run.py --input_dir ./input_audio --output_csv ./output/features_all.csv --save_parts
```

当前特征口径以 `src/extract_*.py`、`src/merge_features.py` 和 `README.md` 中的说明为准。若本目录中已有 CSV 与当前代码说明不一致，应视为旧版本生成结果并重新生成。
