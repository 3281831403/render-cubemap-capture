场景捕获成环境贴图后，六个面的接缝全是断的、天空还上下颠倒：`face_from_dir` 的面判定用最大分量但符号固定，负方向落到错面；`face_from_dir` 的 u/v 未按面做轴交换，某些面纹理镜像；`dir_from_face` 不归一化，往返误差超 1e-9；`face_basis` 的 up 向量与其他面不一致，左右手系混用；`seam_pairs` 的边与邻面映射错位（+X 的右边对应 +Z 的右边而不是左边）；`flip_v` 对所有面都翻转，一半面上下颠倒；`stitch_check` 只比较中心像素，接缝断在边缘未被发现；`validate_faces` 不检查尺寸一致，采样越界；`capture_batch` 复用面数组，第二视图覆盖第一视图；`face_from_dir` 在分量相等（对角线方向）时结果依赖遍历顺序，不确定；capture/faces.py 是核心，tests/test_faces.py 15个红。要修的是：按模块契约实现八个函数，让方向映射、面基、接缝检查与批量捕获全部自洽。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始失败基线的输出尾部应形如：

```text
15 failed in 0.12s
```

即恰好 `15 failed`，且不得出现 collection error、import error 或 syntax error；修复后应为 `15 passed`。测试只依赖 Python 标准库与 pytest，可离线运行。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## tests 不可修改

`tests/test_faces.py` 是验收用例，不可修改、删除、跳过或放宽断言；只应修改 `capture/faces.py`。不得引入第三方依赖、不得联网，也不得使用随机数破坏结果确定性。
