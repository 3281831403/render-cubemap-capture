"""render-cubemap-capture 的 15 条红测。

每条测试都至少调用一个 ``capture.faces`` 中的公开桩。初始桩统一抛
``NotImplementedError``，因此应恰好得到 15 failed，不能出现收集错误。
"""

import math

import pytest

from capture.faces import (
    capture_batch,
    dir_from_face,
    face_basis,
    face_from_dir,
    flip_v,
    seam_pairs,
    stitch_check,
    validate_faces,
)

TOL = 1e-9


def _vec_close(got, want, tol=TOL):
    assert len(got) == len(want)
    for got_value, want_value in zip(got, want):
        assert got_value == pytest.approx(want_value, abs=tol)


def _face(size=2, value=0.0):
    return [[float(value) for _ in range(size)] for _ in range(size)]


def _faces(size=2, value=0.0):
    return [_face(size, value) for _ in range(6)]


def _assert_images_close(got, want, tol=TOL):
    assert len(got) == len(want)
    for got_row, want_row in zip(got, want):
        assert len(got_row) == len(want_row)
        for got_value, want_value in zip(got_row, want_row):
            assert got_value == pytest.approx(want_value, abs=tol)


def _edge_direction(face, edge, s):
    """沿面的指定边取三维方向，参数 s 在 [-1, 1] 内。"""

    right, up, front = face_basis(face)
    if edge == "right":
        base = tuple(front[i] + right[i] for i in range(3))
        axis = up
    elif edge == "left":
        base = tuple(front[i] - right[i] for i in range(3))
        axis = up
    elif edge == "top":
        base = tuple(front[i] + up[i] for i in range(3))
        axis = right
    elif edge == "bottom":
        base = tuple(front[i] - up[i] for i in range(3))
        axis = right
    else:
        raise AssertionError("unknown edge")
    direction = tuple(base[i] + s * axis[i] for i in range(3))
    length = math.sqrt(sum(component * component for component in direction))
    return tuple(component / length for component in direction)


# 组合测试 1：六个轴向方向必须映射到正确的面。
def test_axis_directions_map_to_correct_face():
    axes = [
        ((1.0, 0.0, 0.0), 0),
        ((-1.0, 0.0, 0.0), 1),
        ((0.0, 1.0, 0.0), 2),
        ((0.0, -1.0, 0.0), 3),
        ((0.0, 0.0, 1.0), 4),
        ((0.0, 0.0, -1.0), 5),
    ]
    for direction, want_face in axes:
        face, u, v = face_from_dir(direction)
        assert face == want_face
        assert u == pytest.approx(0.0, abs=TOL)
        assert v == pytest.approx(0.0, abs=TOL)


# 缺陷 1：面判定只看最大分量的大小，符号固定，负方向落到错面。
def test_negative_components_select_negative_faces():
    cases = [
        ((-1.0, 0.2, -0.1), 1),
        ((0.2, -1.0, 0.3), 3),
        ((-0.2, 0.3, -1.0), 5),
    ]
    for direction, want_face in cases:
        face, _, _ = face_from_dir(direction)
        assert face == want_face


# 缺陷 2：u/v 没按面做轴交换，某些面的纹理镜像。
def test_face_uv_uses_face_local_axes():
    cases = [
        ((1.0, 3.0, 2.0), 2, 1.0 / 3.0, -2.0 / 3.0),  # +Y: u=x/|y|, v=-z/|y|
        ((2.0, 1.0, -3.0), 5, 2.0 / 3.0, -1.0 / 3.0),  # -Z: u=x/|z|, v=-y/|z|
        ((4.0, -1.0, 2.0), 0, 0.5, 0.25),            # +X: u=z/|x|, v=-y/|x|
        ((-4.0, -1.0, 2.0), 1, -0.5, 0.25),          # -X: u=-z/|x|, v=-y/|x|
    ]
    for direction, want_face, want_u, want_v in cases:
        face, u, v = face_from_dir(direction)
        assert face == want_face
        assert u == pytest.approx(want_u, abs=TOL)
        assert v == pytest.approx(want_v, abs=TOL)


# 缺陷 3：dir_from_face 不归一化，往返误差超过 1e-9。
def test_dir_from_face_returns_unit_direction():
    samples = [
        (0, 0.6, -0.3),
        (2, -0.5, 0.25),
        (5, 0.9, 0.1),
    ]
    for face, u, v in samples:
        direction = dir_from_face(face, u, v)
        length = math.sqrt(sum(component * component for component in direction))
        assert length == pytest.approx(1.0, abs=TOL)


# 缺陷 4：某个面的 up 与其他面不一致，左右手系混用。
def test_face_basis_up_vectors_are_consistent():
    expected_up = {
        0: (0.0, -1.0, 0.0),
        1: (0.0, -1.0, 0.0),
        2: (0.0, 0.0, -1.0),
        3: (0.0, 0.0, 1.0),
        4: (0.0, -1.0, 0.0),
        5: (0.0, -1.0, 0.0),
    }
    for face, want_up in expected_up.items():
        _, up, front = face_basis(face)
        _vec_close(up, want_up)
        assert sum(up[i] * front[i] for i in range(3)) == pytest.approx(0.0, abs=TOL)


# 缺陷 5：seam_pairs 的边与邻面映射错位。
def test_seam_pairs_maps_each_edge_to_coincident_edge():
    expected = {
        0: [
            ("right", 4, "left"),
            ("top", 3, "right"),
            ("left", 5, "right"),
            ("bottom", 2, "right"),
        ],
        1: [
            ("right", 5, "left"),
            ("top", 3, "left"),
            ("left", 4, "right"),
            ("bottom", 2, "left"),
        ],
        2: [
            ("right", 0, "bottom"),
            ("top", 5, "bottom"),
            ("left", 1, "bottom"),
            ("bottom", 4, "bottom"),
        ],
        3: [
            ("right", 0, "top"),
            ("top", 4, "top"),
            ("left", 1, "top"),
            ("bottom", 5, "top"),
        ],
        4: [
            ("right", 1, "left"),
            ("top", 3, "top"),
            ("left", 0, "right"),
            ("bottom", 2, "bottom"),
        ],
        5: [
            ("right", 0, "left"),
            ("top", 3, "bottom"),
            ("left", 1, "right"),
            ("bottom", 2, "top"),
        ],
    }
    for face in range(6):
        assert seam_pairs(face) == expected[face]


# 缺陷 6：flip_v 对所有面都翻转，一半的面上下颠倒。
def test_flip_v_flips_only_positive_faces():
    source = [[1.0, 2.0], [3.0, 4.0]]
    flipped = [[3.0, 4.0], [1.0, 2.0]]

    for face in (0, 2, 4):
        result = flip_v(face, source)
        assert result == flipped
        assert result is not source
        assert result[0] is not source[0]

    for face in (1, 3, 5):
        result = flip_v(face, source)
        assert result == source
        assert result is not source
        assert result[0] is not source[0]
        result[0][0] = 99.0
        assert source[0][0] == pytest.approx(1.0, abs=TOL)

    assert source == [[1.0, 2.0], [3.0, 4.0]]


# 缺陷 7：stitch_check 只看中心像素，边缘断裂漏检。
def test_stitch_check_detects_edge_discontinuity():
    faces = _faces(4, 0.0)
    faces[0][1][3] = 1.0  # +X 面右边缘、非中心的一个像素

    differences = stitch_check(faces)

    assert isinstance(differences, list)
    assert len(differences) == 24
    assert all(isinstance(value, (int, float)) for value in differences)
    assert all(math.isfinite(float(value)) for value in differences)
    # index 0 是 +X 的 right 边；断裂必须在整条边上被发现。
    assert differences[0] > 0.9


# 缺陷 8：validate_faces 不检查尺寸一致，后续采样越界。
def test_validate_faces_flags_structure_size_and_nan():
    assert any(
        problem.startswith("count:") for problem in validate_faces(_faces(2)[:5])
    )

    mismatched = _faces(2, 0.1)
    mismatched[5] = _face(3, 0.1)
    problems = validate_faces(mismatched)
    assert problems
    assert any(problem.startswith("size:") for problem in problems)
    with pytest.raises(ValueError):
        stitch_check(mismatched)

    nan_faces = _faces(2, 0.1)
    nan_faces[1][0][1] = float("nan")
    problems = validate_faces(nan_faces)
    assert problems
    assert any(problem.startswith("nan:") for problem in problems)

    assert validate_faces(_faces(2, 0.1)) == []


# 缺陷 9：capture_batch 复用面数组，第二视图覆盖第一视图。
def test_capture_batch_keeps_views_independent():
    def sampler(view, face, u, v):
        return view["value"] + 0.125 * face + 0.01 * u + 0.001 * v

    views = [{"id": "left", "value": 1.0}, {"id": "right", "value": 7.0}]
    config = {"size": 2, "sampler": sampler}

    result = capture_batch(views, config)

    assert list(result) == ["left", "right"]
    # N=2 时左上角纹素中心为 (u, v) = (-0.5, 0.5)。
    assert result["left"][0][0][0] == pytest.approx(
        sampler(views[0], 0, -0.5, 0.5), abs=TOL
    )
    assert result["right"][0][0][0] == pytest.approx(
        sampler(views[1], 0, -0.5, 0.5), abs=TOL
    )
    assert result["left"][0][0][0] != pytest.approx(
        result["right"][0][0][0], abs=TOL
    )
    for face in range(6):
        assert result["left"][face] is not result["right"][face]
        for row in range(2):
            assert result["left"][face][row] is not result["right"][face][row]


# 缺陷 10：分量相等时按遍历顺序选面，结果不确定。
def test_tied_directions_use_axis_priority():
    cases = [
        ((1.0, 1.0, 0.0), 0, 0.0, -1.0),
        ((-1.0, -1.0, 0.0), 1, 0.0, 1.0),
        ((0.0, 1.0, 1.0), 2, 0.0, -1.0),
        ((1.0, 0.0, 1.0), 0, 1.0, 0.0),
        ((0.0, -1.0, 1.0), 3, 0.0, 1.0),
    ]
    for direction, want_face, want_u, want_v in cases:
        face, u, v = face_from_dir(direction)
        assert face == want_face
        assert u == pytest.approx(want_u, abs=TOL)
        assert v == pytest.approx(want_v, abs=TOL)
        second = face_from_dir(direction)
        assert second[0] == want_face
        assert second[1] == pytest.approx(want_u, abs=TOL)
        assert second[2] == pytest.approx(want_v, abs=TOL)


# 组合测试 2：方向 -> 面 -> 方向往返误差小于 1e-9。
def test_direction_roundtrips_through_face():
    samples = [
        (0, 0.35, -0.45),
        (1, -0.2, 0.7),
        (2, 0.55, 0.15),
        (3, -0.65, -0.25),
        (4, 0.1, -0.8),
        (5, -0.4, 0.5),
    ]
    for face, u, v in samples:
        direction = dir_from_face(face, u, v)
        got_face, got_u, got_v = face_from_dir(direction)
        assert got_face == face
        assert got_u == pytest.approx(u, abs=TOL)
        assert got_v == pytest.approx(v, abs=TOL)


# 组合测试 3：接缝两侧取出的是同一条三维边，方向连续。
def test_seams_are_continuous_on_both_sides():
    parameters = (-0.8, -0.35, 0.0, 0.3, 0.85)
    for face in range(6):
        pairs = seam_pairs(face)
        assert len(pairs) == 4
        for edge, neighbor, neighbor_edge in pairs:
            left = [_edge_direction(face, edge, s) for s in parameters]
            right = [_edge_direction(neighbor, neighbor_edge, s) for s in parameters]
            # 邻边的参数方向可以与本边一致或相反，两种朝向都必须落在同一条棱上。
            right += [_edge_direction(neighbor, neighbor_edge, -s) for s in parameters]
            for left_point in left:
                nearest = min(
                    math.dist(left_point, right_point) for right_point in right
                )
                assert nearest < TOL


# 组合测试 4：六个面的基向量必须构成右手系。
def test_face_bases_form_right_handed_system():
    for face in range(6):
        right, up, front = face_basis(face)
        for vector in (right, up, front):
            length = math.sqrt(sum(component * component for component in vector))
            assert length == pytest.approx(1.0, abs=TOL)
        assert sum(right[i] * up[i] for i in range(3)) == pytest.approx(0.0, abs=TOL)
        assert sum(right[i] * front[i] for i in range(3)) == pytest.approx(
            0.0, abs=TOL
        )
        assert sum(up[i] * front[i] for i in range(3)) == pytest.approx(0.0, abs=TOL)
        cross = (
            right[1] * up[2] - right[2] * up[1],
            right[2] * up[0] - right[0] * up[2],
            right[0] * up[1] - right[1] * up[0],
        )
        _vec_close(cross, front)


# 组合测试 5：两视图批量捕获必须与各自单独捕获完全一致。
def test_two_views_match_individual_captures():
    def sampler(view, face, u, v):
        return view["bias"] + 0.01 * face + 0.1 * u - 0.05 * v

    views = [
        {"id": "front", "bias": 0.25},
        {"id": "back", "bias": 2.5},
    ]
    config = {"size": 3, "sampler": sampler}

    batch = capture_batch(views, config)

    for view in views:
        single = capture_batch([view], config)
        assert list(single) == [view["id"]]
        _assert_images_close(batch[view["id"]], single[view["id"]])
