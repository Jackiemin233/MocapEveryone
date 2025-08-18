# 多进程加速处理说明

## 概述

本脚本已添加多进程支持，可以显著加速数据处理过程。通过并行处理多个文件，可以充分利用多核CPU资源。

## 使用方法

### 基本用法

```bash
python eval_by_files_train.py \
    --base-dir /path/to/data \
    --data-config-path /path/to/config \
    --save-path /path/to/save \
    --data-type train \
    --setting vr \
    --num-processes 8
```

### 参数说明

- `--num-processes`: 指定使用的进程数量
  - 默认值：自动设置为CPU核心数的75%
  - 建议值：根据你的系统配置调整
  - 示例：`--num-processes 4` 使用4个进程

### 自动进程数设置

如果不指定 `--num-processes` 参数，脚本会自动检测系统CPU核心数并设置为75%：

```bash
# 自动设置进程数
python eval_by_files_train.py \
    --base-dir /path/to/data \
    --data-config-path /path/to/config \
    --save-path /path/to/save \
    --data-type train \
    --setting vr
```

## 性能提升

- **单进程**: 按顺序处理文件，适合调试
- **多进程**: 并行处理文件，显著提升速度
- **预期提升**: 在8核CPU上可达到6-7倍速度提升

## 系统兼容性

### macOS
- 自动使用 `spawn` 启动方法
- 避免 `fork` 方法可能的问题

### Linux/Windows
- 使用系统默认的多进程启动方法

## 错误处理

- 如果多进程失败，会自动回退到单进程模式
- 提供详细的错误信息和回退提示

## 测试多进程功能

运行测试脚本验证多进程功能：

```bash
python test_multiprocessing.py
```

## 注意事项

1. **内存使用**: 多进程会增加内存使用量，确保系统有足够内存
2. **文件I/O**: 多进程同时读写文件时，确保文件系统支持并发访问
3. **调试**: 调试时建议使用单进程模式 (`--num-processes 1`)
4. **进程数选择**: 
   - 太少：无法充分利用CPU资源
   - 太多：可能导致系统过载，建议不超过CPU核心数

## 故障排除

### 常见问题

1. **进程启动失败**
   - 检查系统资源（内存、文件描述符限制）
   - 减少进程数量

2. **性能提升不明显**
   - 检查是否为I/O密集型任务
   - 调整进程数量

3. **内存不足**
   - 减少进程数量
   - 检查系统内存使用情况

### 调试模式

```bash
# 强制使用单进程进行调试
python eval_by_files_train.py \
    --base-dir /path/to/data \
    --data-config-path /path/to/config \
    --save-path /path/to/save \
    --data-type train \
    --setting vr \
    --num-processes 1
``` 