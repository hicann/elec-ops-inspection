# operators 目录

`operators/` 是电力巡检算子实现的统一入口。后续新增算子请放在 `operators/<operator_name>/` 下，避免根目录随着算子数量增长而变得分散。

## 新增算子放置方式

单个算子建议使用以下结构：

```text
operators/<operator_name>/
├── README.md
├── op_host/
├── op_kernel/
├── examples/
├── tests/
├── docs/
└── scripts/
```

其中只有 `README.md` 是建议必备项，其余目录按算子实际内容创建。算子内部的构建脚本、样例和说明文档应优先留在该算子目录内；跨算子的公共脚本或场景示例再放到仓库级目录。

## 与历史目录的关系

仓库当前已有部分算子位于根目录，这是历史路径。新增算子请使用 `operators/` 目录；历史算子如需迁移，应单独提交 PR，并在 PR 中说明兼容影响。
