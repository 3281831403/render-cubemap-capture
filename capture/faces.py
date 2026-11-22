"""立方体环境贴图捕获。

坐标与数据约定
--------------
* 世界坐标为右手系：``+X`` 向右、``+Y`` 向上、``+Z`` 指向观察者。
* 面索引固定为 ``0=+X、1=-X、2=+Y、3=-Y、4=+Z、5=-Z``。
* 每个面使用右手基 ``(right, up, front)``，满足
  ``cross(right, up) == front``；``front`` 是该面的外法线，``right``
  是面内 u 增大的方向，``up`` 是面内 v 增大的方向。
* 面内坐标 ``u、v`` 均位于 ``[-1, 1]``，面中心为 ``(0, 0)``。
* 面图像是 ``N × N`` 的二维列表 ``image[row][col]``。图像第 0 行是
  顶部，对应 ``v = +1``；第 0 列是左侧，对应 ``u = -1``。第 ``row``
  行、第 ``col`` 列纹素的中心为
  ``u = -1 + (2*col + 1)/N``、``v = +1 - (2*row + 1)/N``。
* 所有浮点结果的比较容差为 ``1e-9``；模块只使用 Python 标准库，
  不联网、不使用随机数，结果完全确定。
"""

from __future__ import annotations

import math

FACE_COUNT = 6
FACE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
EDGE_ORDER = ("right", "top", "left", "bottom")
TOLERANCE = 1e-9

_BASIS = (
    ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
)

# 接缝表：seam_pairs(face)[i] 对应 EDGE_ORDER[i] 的 (邻面, 邻边)。
# 每条邻边都是三维上与本边重合的那条立方体棱。
_SEAMS = (
    ((4, "left"), (3, "right"), (5, "right"), (2, "right")),
    ((5, "left"), (3, "left"), (4, "right"), (2, "left")),
    ((0, "bottom"), (5, "bottom"), (1, "bottom"), (4, "bottom")),
    ((0, "top"), (4, "top"), (1, "top"), (5, "top")),
    ((1, "left"), (3, "top"), (0, "right"), (2, "bottom")),
    ((0, "left"), (3, "bottom"), (1, "right"), (2, "top")),
)

# 捕获后端对正面向返回 D3D 行序（第 0 行对应 v=-1），需要翻转。
_FLIP_V_FACES = (0, 2, 4)


def _check_face_index(face) -> int:
    if isinstance(face, bool) or not isinstance(face, int):
        raise ValueError(f"面索引必须是 0..5 的整数，得到 {face!r}")
    if not 0 <= face < FACE_COUNT:
        raise ValueError(f"面索引越界：{face}，合法范围为 0..5")
    return face


def _check_finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限实数，得到 {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限实数，得到 {value!r}")
    return value


def face_from_dir(
    direction: tuple[float, float, float],
) -> tuple[int, float, float]:
    """把世界方向映射为 ``(face, u, v)``。

    按绝对值最大的主轴选面，符号由该分量的符号决定；主轴并列时
    使用固定优先级 ``X > Y > Z``。面内坐标用该面的右手基投影：
    ``u = dot(direction, right) / m``、``v = dot(direction, up) / m``，
    其中 ``m`` 是最大分量的绝对值。
    """

    try:
        x, y, z = direction
    except (TypeError, ValueError) as exc:
        raise ValueError(f"方向必须是三个有限实数，得到 {direction!r}") from exc
    x = _check_finite(x, "direction[0]")
    y = _check_finite(y, "direction[1]")
    z = _check_finite(z, "direction[2]")

    ax, ay, az = abs(x), abs(y), abs(z)
    m = max(ax, ay, az)
    if m == 0.0:
        raise ValueError("方向不能是零向量")

    # 固定优先级 X > Y > Z：先 |x| 与 |y|，再 |y| 与 |z|。
    if ax >= ay and ax >= az:
        face = 0 if x > 0.0 else 1
    elif ay >= az:
        face = 2 if y > 0.0 else 3
    else:
        face = 4 if z > 0.0 else 5

    right, up, _ = _BASIS[face]
    u = (x * right[0] + y * right[1] + z * right[2]) / m
    v = (x * up[0] + y * up[1] + z * up[2]) / m
    return face, u, v


def dir_from_face(face: int, u: float, v: float) -> tuple[float, float, float]:
    """把面内坐标 ``(face, u, v)`` 还原为单位世界方向。

    ``d = front + u * right + v * up``，再归一化到单位球面。
    """

    _check_face_index(face)
    u = _check_finite(u, "u")
    v = _check_finite(v, "v")

    right, up, front = _BASIS[face]
    d = tuple(front[i] + u * right[i] + v * up[i] for i in range(3))
    length = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    return (d[0] / length, d[1] / length, d[2] / length)


def face_basis(face: int) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """返回面的 ``(right, up, front)`` 右手基。"""

    _check_face_index(face)
    return _BASIS[face]


def seam_pairs(face: int) -> list[tuple[str, int, str]]:
    """返回一个面四条边对应的邻面与邻边。

    顺序固定为 ``("right", "top", "left", "bottom")``，每个元素是
    ``(edge, neighbor_face, neighbor_edge)``，邻边是三维上重合的棱。
    """

    _check_face_index(face)
    return [
        (edge, neighbor, neighbor_edge)
        for edge, (neighbor, neighbor_edge) in zip(EDGE_ORDER, _SEAMS[face])
    ]


def flip_v(face: int, image: list[list[float]]) -> list[list[float]]:
    """按面的朝向翻转图像的 v 方向，且不修改输入。

    正面向 ``0=+X、2=+Y、4=+Z`` 反转行顺序；负面向保持行顺序。
    无论是否反转，外层列表与每一行都是新对象。
    """

    _check_face_index(face)
    rows = reversed(image) if face in _FLIP_V_FACES else image
    return [list(row) for row in rows]


def validate_faces(faces: list[list[list[float]]]) -> list[str]:
    """校验六个面是否完整、同尺寸且不含 NaN。

    返回以 ``count:``、``shape:``、``size:`` 或 ``nan:`` 开头的问题
    描述列表；合法输入返回空列表。任何非法输入都不抛出异常。
    """

    problems: list[str] = []

    try:
        count = len(faces)
    except TypeError:
        return [f"count: faces 不是长度为 6 的序列，得到 {type(faces).__name__}"]
    if count != FACE_COUNT:
        problems.append(f"count: 面数为 {count}，应为 {FACE_COUNT}")

    sizes: list[int] = []
    for index in range(min(count, FACE_COUNT)):
        try:
            face = faces[index]
        except (IndexError, KeyError, TypeError) as exc:
            problems.append(f"shape: 面 {index} 无法读取（{exc}）")
            continue
        try:
            rows = [list(row) for row in face]
        except TypeError:
            problems.append(f"shape: 面 {index} 不是二维行序列")
            continue
        if not rows:
            problems.append(f"shape: 面 {index} 为空")
            continue
        size = len(rows)
        bad_shape = False
        for row in rows:
            if len(row) != size:
                problems.append(
                    f"shape: 面 {index} 不是 {size}×{size} 的正方形矩形图像"
                )
                bad_shape = True
                break
        if bad_shape:
            continue
        sizes.append(size)
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    problems.append(
                        f"nan: 面 {index} 像素 ({row_index}, {col_index}) "
                        f"不是有限实数：{value!r}"
                    )

    if count == FACE_COUNT and len(sizes) == FACE_COUNT and len(set(sizes)) != 1:
        problems.append(f"size: 六个面尺寸不一致：{sizes}")

    return problems


def _edge_samples(face: int, edge: str, image: list[list[float]]):
    """沿面的一条边逐像素取 ``(方向, 像素值)``，参数 s 从 -1 到 1。"""

    size = len(image)
    samples = []
    for i in range(size):
        s = -1.0 + (2 * i + 1) / size
        if edge == "right":
            u, v = 1.0, s
            row, col = size - 1 - i, size - 1
        elif edge == "left":
            u, v = -1.0, s
            row, col = size - 1 - i, 0
        elif edge == "top":
            u, v = s, 1.0
            row, col = 0, i
        else:  # bottom
            u, v = s, -1.0
            row, col = size - 1, i
        samples.append((dir_from_face(face, u, v), image[row][col]))
    return samples


def stitch_check(faces: list[list[list[float]]]) -> list[float]:
    """检查六个面在接缝处的像素差。

    先按 :func:`validate_faces` 校验输入；再对每条有向边沿整条边
    逐像素与邻面对应边比较（按三维端点方向对齐正序或逆序），返回
    24 个最大绝对差，顺序为 ``face * 4 + edge_index``。
    """

    problems = validate_faces(faces)
    if problems:
        raise ValueError("面数据不合法：" + "; ".join(problems))

    size = len(faces[0])
    differences: list[float] = []
    for face in range(FACE_COUNT):
        for edge, neighbor, neighbor_edge in seam_pairs(face):
            own = _edge_samples(face, edge, faces[face])
            other = _edge_samples(neighbor, neighbor_edge, faces[neighbor])

            # 端点方向距离决定两条边的像素顺序如何对齐。
            forward = math.dist(own[0][0], other[0][0]) + math.dist(
                own[-1][0], other[-1][0]
            )
            backward = math.dist(own[0][0], other[-1][0]) + math.dist(
                own[-1][0], other[0][0]
            )
            if backward < forward:
                other = other[::-1]

            worst = 0.0
            for i in range(size):
                diff = abs(own[i][1] - other[i][1])
                if diff > worst:
                    worst = diff
            differences.append(worst)
    return differences


def capture_batch(
    views: list[dict],
    config: dict,
) -> dict:
    """批量捕获多个独立视图的六面贴图。

    对每个视图、每个面、每个纹素中心调用一次
    ``config["sampler"](view, face, u, v)``，返回
    ``{view["id"]: [face_0, ..., face_5]}``。每个视图的二维面数组与
    行列表完全独立分配。
    """

    try:
        size = config["size"]
        sampler = config["sampler"]
    except (TypeError, KeyError) as exc:
        raise ValueError(f"config 缺少必要键：{exc}") from exc
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"size 必须是正整数，得到 {size!r}")
    if not callable(sampler):
        raise ValueError("sampler 必须是可调用对象")

    result: dict = {}
    for view in views:
        try:
            view_id = view["id"]
        except (TypeError, KeyError) as exc:
            raise ValueError(f"视图缺少 \"id\" 键：{view!r}") from exc
        try:
            hash(view_id)
        except TypeError as exc:
            raise ValueError(f"视图 id 不可哈希：{view_id!r}") from exc
        if view_id in result:
            raise ValueError(f"视图 id 重复：{view_id!r}")

        face_images = []
        for face in range(FACE_COUNT):
            image = []
            for row in range(size):
                v = 1.0 - (2 * row + 1) / size
                line = []
                for col in range(size):
                    u = -1.0 + (2 * col + 1) / size
                    line.append(float(sampler(view, face, u, v)))
                image.append(line)
            face_images.append(image)
        result[view_id] = face_images
    return result
