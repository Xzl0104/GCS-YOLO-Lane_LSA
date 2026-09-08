# CULane Training And Evaluation

The converted CULane dataset is stored at:

```text
D:\gcsxxl\GCS-YOLO-Lane_LSA_5-25-3-k56\datasets\culane_fixed_y_590x960
```

It uses the GCS fixed-y K56 contract with input shape `H,W=(384,960)`.
Every lane is resampled at the shared original-image anchors
`589, 579, ..., 39` on CULane's `H,W=(590,1640)` images. CULane labels may
contain `0..4` lanes, including valid zero-lane images. The original CULane
annotations remain under:

```text
D:\BaiduNetdiskDownload\CULane
```

## Evaluation Metric

Use `tools/eval_gcs.py --metric culane_iou` for CULane benchmark-style
evaluation. Predictions and original `.lines.txt` annotations are mapped to
the original `W,H=(1640,590)` coordinate system. Each lane is rasterized as a
30-pixel-wide region, matched with Hungarian assignment, and counted as a
true positive only when `IoU > 0.5`.

The existing `--metric ape` mode remains available for geometry diagnostics
and backward compatibility, but its F1 must not be reported as the CULane
benchmark F1.

## Evaluation Command

After a checkpoint has been selected from the validation split, run:

```powershell
& 'D:\miniconda3\envs\lsa_yolo\python.exe' tools/eval_gcs.py `
  --dataset culane `
  --weights '<path-to-selected-checkpoint.pt>' `
  --source 'D:\gcsxxl\GCS-YOLO-Lane_LSA_5-25-3-k56\datasets\culane_fixed_y_590x960\images\test' `
  --labels 'D:\gcsxxl\GCS-YOLO-Lane_LSA_5-25-3-k56\datasets\culane_fixed_y_590x960\labels_gcs\test' `
  --imgsz 384 960 `
  --metric culane_iou `
  --culane-archive-root 'D:\BaiduNetdiskDownload\CULane' `
  --device 0 `
  --save-json `
  --save-dir '<output-dir>\culane_test_eval'
```

Use the validation split for checkpoint and decoder selection. Run the final
test evaluation only after that selection is frozen.

## Result Table

Generate the table from an evaluation summary containing
`metric_name=culane_iou`:

```powershell
& 'D:\miniconda3\envs\lsa_yolo\python.exe' tools/summarize_culane_table.py `
  --eval-summary '<output-dir>\culane_test_eval\eval_summary.json' `
  --manifest 'D:\gcsxxl\GCS-YOLO-Lane_LSA_5-25-3-k56\datasets\culane_fixed_y_590x960\manifests\manifest.json' `
  --method 'GCS-YOLO-Lane' 
```

The generated files are:

```text
culane_table.csv
culane_table.md
culane_table.tex
culane_table_details.json
```

`Normal`, `Crowded`, `Dazzle`, `Shadow`, `No line`, `Arrow`, `Curve`, and
`Night` are F1 percentages. `Crossroad` follows the CULane paper table
convention and reports the false-positive count. `Total` is the overall F1
percentage, and `FPS` uses inference plus post-processing time.
