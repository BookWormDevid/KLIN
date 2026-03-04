# Triton Model Repository

This directory is mounted into Triton as `/models` by `docker-compose.infra.yml`.

## Layout

```text
model_repository/
  yolo_person/
    config.pbtxt
    1/
      .gitkeep            # replace with model.onnx
  x3d_violence/
    config.pbtxt
    1/
      .gitkeep            # replace with model.onnx
  videomae_crime/
    config.pbtxt
    1/
      .gitkeep            # replace with model.onnx
```

## Important

- Replace each `.gitkeep` with actual `model.onnx`.
- If ONNX input/output names differ from `config.pbtxt`, update config to match.
- Class/threshold logic remains in application code, not in Triton configs.
