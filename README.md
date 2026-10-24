# AI6103：AutoDL 一键实验

使用 torchvision EfficientNetV2-S 从零训练，完成作业的全部 8 组实验、
训练/验证曲线、Beta 分布图、最终测试，以及英文 AAAI 双栏报告。

## 在 AutoDL 运行

先激活你已经安装 CUDA PyTorch 的 Python 环境，然后执行：

```bash
cd /root/autodl-tmp
git clone https://github.com/yichuishan/AI6103-Assignment.git
cd AI6103-Assignment
bash run_autodl.sh /root/autodl-tmp/data
```

把最后一行的数据路径换成实际 ZIP 或已解压目录。如果文件正好是
`/root/autodl-tmp/data.zip`，也可以直接 `bash run_autodl.sh`。
如果仓库已克隆，在仓库内先 `git pull --ff-only`，再执行运行命令。

脚本自动完成：

1. 检查并补齐 Python 依赖和 PDF 编译工具。
2. 识别数据划分，检查原始 7,155/1,533/1,534 张和 120 类，再剔除重复与冲突标注记录。
3. 只用清洗后的训练集重新计算 RGB 均值和标准差，检查 batch size 128 的 GPU 前后向运行。
4. 验证 PDF 工具、运行测试，然后在后台执行 **8 组、共 1,545 epochs**。
5. 根据验证集选择模型，最后评估测试集，生成全部图表、数值分析和报告。
6. 输出结果压缩包。全部文件只保存在 AutoDL 服务器上。

后台启动后可以关闭网页终端。训练耗时取决于 GPU 和数据读取速度，脚本会逐
epoch 记录耗时；不要将缩短的功能测试结果用于作业。中断后执行**同一命令**
即可从完整 epoch 恢复；运行中再次启动会被锁阻止。

## 查看和下载结果

```bash
tail -f results_cleaned/pipeline.log
cat results_cleaned/status.json
```

- **`results_cleaned/AI6103_results.zip`**：最终结果包，包含报告、图表、指标和日志数据。
- **`results_cleaned/report/report.pdf`**：英文报告。
- `results_cleaned/report/generated/results.csv`：各条件最终训练/验证 loss 和 accuracy。
- `results_cleaned/artifacts/figures/`：全部 PDF/PNG 图，含独立大图和报告用图。
- `results_cleaned/runs/`：每组逐 epoch CSV、检查点和选择记录；模型文件不放入结果包。

`status.json` 的 `state` 为 `complete` 才表示整个流程成功。错误会进入
`failed`，详细原因见终端或日志。未完成的真实实验不会被填成估计结果。
报告正文超过 6 页会报错，脚本不会修改 AAAI 字号或页边距。
提交前请检查结果与论述，并补上个人作者信息。

## 数据格式与选项

优先识别现成的 `train/<breed>/`、`val/<breed>/`、`test/<breed>/`，
支持外层嵌套目录；验证目录也可叫 `valid` 或 `validation`。
还支持 `path,label,split` CSV，以及常见的 train/val/test 独立标签 CSV。
作业数据的 `train_labels.csv`、`val_labels.csv`、`test_labels.csv` 配合
`train/images/`、`val/images/`、`test/images/` 可直接识别，标签取自 CSV，保留现成划分。
已按实际摘要确认：CSV 为 `id,breed`，`id` 不带扩展名，图片文件为 `<id>.jpg`；
三个集合均包含 120 类。测试包含与实际图片数、标签行数和类别数量相同的
10,222 条格式夹具，并检查 CSV 行顺序与文件名排序不同时仍正确匹配标签。
夹具不含真实图片，其生成的中间类别分布不代表原始数据分布。
只有原始 `labels.csv` + `train/<id>.jpg`、没有现成划分时，才生成并记录
新的固定种子精确数量分层划分，不把它冒充为官方划分。
其他文件结构可显式提供 `--manifest /absolute/path/to/manifest.csv`。

训练前按以下规则清洗全部原始记录：

- 文件 SHA-256 相同后，再逐字节确认内容一致。
- 同一内容对应不同标签：整组剔除，不任选一个标签。
- 标签一致的重复组：只保留一份；跨集合优先保留测试集，其次验证集、训练集，同集合内按路径排序选择。
- 只从实验清单中排除，不删除或覆盖原始图片、CSV；不把图片转移到其他集合。
- 原始数量检查仍保留；清洗后的数量按实际保留记录计算，不补样凑数。
- 清洗后再次检查无字节重复／标签冲突，确认训练类别仍完整，再重算训练集均值和标准差。

`results_cleaned/cleaning_summary.json` 保存原始／保留／剔除数量和原因，
`removed_images.csv` 列出每个剔除文件，`cleaned_manifest.csv` 是实际用于训练、验证和测试的清单。
原始质量审计包含所有重复组（含集合内部重复），保存在 `data_quality_audit.json`。
这些文件会纳入最终结果包，报告的预处理部分自动列出清洗规则和实际样本数。

重新训练默认写入 **`results_cleaned/`**，旧的 `results/` 保留。
清洗策略加入数据指纹和运行配置，旧 prepared 文件和检查点不能用于本次恢复。
本次实际清洗后的测试集与之前的原始测试集不同，前后分数的样本范围也不同。

```bash
# 作者姓名会写入英文报告：
bash run_autodl.sh /path/to/data.zip --author "Your Name"
# 前台运行：
bash run_autodl.sh /path/to/data.zip --foreground
# 只安装、检查和准备数据，不启动全部训练：
bash run_autodl.sh /path/to/data.zip --check-only
# 指定新结果目录或使用全精度：
bash run_autodl.sh /path/to/data.zip --output /root/autodl-tmp/another-run --precision fp32
```

默认在支持时使用 BF16，否则 FP16，所有条件精度一致并记录到报告，评估
采用 FP32。batch size 始终为 128。参见 [EXPERIMENTS.md](EXPERIMENTS.md)
了解选择规则、指标定义、断点恢复和作业要求对应关系。
已安装的 CUDA PyTorch 会被保留，其他依赖由 `requirements.txt` 安装。
初次补齐 TeX 工具需要 root 权限和网络；AutoDL 常规容器满足权限条件。

## 功能验证与手动使用

```bash
python -m pytest -q
# 合成数据 CPU 全流程：明确标为验证用途，不能作为作业结果。
bash run_autodl.sh --smoke --foreground --output /tmp/ai6103-smoke
```

`--smoke` 需要已有 CPU 或 CUDA PyTorch，使用真实网络，但只有 12 张合成图、
每组 2 epochs，并使用适合小批次的较低学习率。正式作业仍使用 0.5/0.005/0.001、
batch 128 和规定的 15/300 epochs。正式报告生成器会拒绝合成统计和结果。

也可用 `python -m ai6103.cli --help` 分别运行 prepare/run/test/plot/report。
报告使用未修改的 AAAI 2024 样式，来源见 [STYLE_SOURCE.md](report/STYLE_SOURCE.md)。
现有云任务已有隔离工作区，无需为安装再创建 Git worktree。
