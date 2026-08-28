"""Unit tests for the NN primitives and feature construction."""

from std.math import pi
from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_true

from trackpermanence.features import LocalFrame, completion_frame, hermite_prior, motion_features, time_queries, wrap_angle
from trackpermanence.nn import gru, layernorm, mha, mlp, sigmoid, softmax_rows_
from trackpermanence.tensor import Tensor
from trackpermanence.types import BoxState


def _fill(mut t: Tensor, v: Float32):
    for i in range(t.numel()):
        t[i] = v


def _eye(n: Int) -> Tensor:
    var t = Tensor([n, n])
    for i in range(n):
        t.set2(i, i, 1.0)
    return t^


def test_sigmoid_and_softmax() raises:
    assert_almost_equal(Float64(sigmoid(0.0)), 0.5, atol=Float64(1e-7))
    var s = Tensor([2, 3])
    s.set2(0, 0, 1.0)
    s.set2(0, 1, 2.0)
    s.set2(0, 2, 3.0)
    s.set2(1, 0, 5.0)
    softmax_rows_(s)
    for i in range(2):
        var total: Float32 = 0.0
        for j in range(3):
            total += s.at2(i, j)
        assert_almost_equal(Float64(total), 1.0, atol=Float64(1e-6))
    assert_true(s.at2(0, 2) > s.at2(0, 1) and s.at2(0, 1) > s.at2(0, 0))


def test_mlp_identity_layers() raises:
    var x = Tensor([2, 3])
    x.set2(0, 0, 1.0)
    x.set2(0, 1, -2.0)
    x.set2(1, 2, 3.0)
    var zero_b = Tensor([3])
    var y = mlp(x, _eye(3), zero_b, _eye(3), zero_b)
    assert_almost_equal(Float64(y.at2(0, 0)), 1.0, atol=Float64(1e-7))
    assert_almost_equal(Float64(y.at2(0, 1)), 0.0, atol=Float64(1e-7))  # relu
    assert_almost_equal(Float64(y.at2(1, 2)), 3.0, atol=Float64(1e-7))


def test_gru_zero_weights_keeps_state() raises:
    # with all-zero weights: r = z = 0.5, n = 0 -> h' = 0.5 * h each step
    var d = 2
    var x = Tensor([3, 2])
    var w_ih = Tensor([3 * d, 2])
    var w_hh = Tensor([3 * d, d])
    var b = Tensor([3 * d])
    var h0 = Tensor([1, d])
    _fill(h0, 1.0)
    var st = gru(x, w_ih, b, w_hh, b, h0, False)
    assert_almost_equal(Float64(st.h[0]), 0.125, atol=Float64(1e-6))
    assert_almost_equal(Float64(st.out.at2(0, 0)), 0.5, atol=Float64(1e-6))
    var rev = gru(x, w_ih, b, w_hh, b, h0, True)
    assert_almost_equal(Float64(rev.out.at2(2, 0)), 0.5, atol=Float64(1e-6))  # first visited row


def test_mha_identity_projections_averages_values() raises:
    # identity in/out projections and equal keys -> uniform attention -> mean of values
    var d = 4
    var x_q = Tensor([1, d])
    var x_kv = Tensor([2, d])
    x_kv.set2(0, 0, 2.0)
    x_kv.set2(1, 0, 4.0)
    var in_w = Tensor([3 * d, d])
    for i in range(3):
        for c in range(d):
            in_w.set2(i * d + c, c, 1.0)
    var in_b = Tensor([3 * d])
    var out_b = Tensor([d])
    var y = mha(x_q, x_kv, in_w, in_b, _eye(d), out_b, 2)
    assert_almost_equal(Float64(y.at2(0, 0)), 3.0, atol=Float64(1e-6))


def test_layernorm_normalizes() raises:
    var x = Tensor([1, 4])
    x.set2(0, 0, 1.0)
    x.set2(0, 1, 2.0)
    x.set2(0, 2, 3.0)
    x.set2(0, 3, 4.0)
    var g = Tensor([4])
    _fill(g, 1.0)
    var y = layernorm(x, g, Tensor([4]))
    var mean: Float32 = 0.0
    for c in range(4):
        mean += y.at2(0, c)
    assert_almost_equal(Float64(mean), 0.0, atol=Float64(1e-5))
    assert_true(y.at2(0, 3) > y.at2(0, 0))


def test_local_frame_roundtrip() raises:
    var fr = LocalFrame(3.0, -2.0, 0.7)
    var l = fr.to_local(1.5, 2.5)
    var g = fr.to_global(l[0], l[1])
    assert_almost_equal(g[0], 1.5, atol=Float64(1e-12))
    assert_almost_equal(g[1], 2.5, atol=Float64(1e-12))
    assert_almost_equal(wrap_angle(3.0 * pi), pi, atol=Float64(1e-12))


def _state(t: Float64, x: Float64, y: Float64, vx: Float64, theta: Float64) -> BoxState:
    return BoxState(frame=0, t=t, x=x, y=y, z=0.0, w=2.0, l=4.0, h=1.5, vx=vx, vy=0.0, theta=theta, conf=0.9, cls=0, tid=1)


def test_motion_features_last_pose_is_origin() raises:
    var states = List[BoxState]()
    states.append(_state(0.0, 0.0, 0.0, 5.0, 0.0))
    states.append(_state(0.5, 2.5, 0.0, 5.0, 0.0))
    var last = states[1]
    var f = motion_features(states, LocalFrame(last.x, last.y, last.theta), last.t)
    assert_equal(f.dim(0), 2)
    assert_equal(f.dim(1), 8)
    for c in range(4):
        assert_almost_equal(Float64(f.at2(1, c)), 0.0, atol=Float64(1e-7))
    assert_almost_equal(Float64(f.at2(0, 0)), -0.25, atol=Float64(1e-6))  # -2.5 m * 0.1
    assert_almost_equal(Float64(f.at2(0, 3)), -0.1, atol=Float64(1e-6))  # -0.5 s * 0.2
    assert_almost_equal(Float64(f.at2(0, 6)), 0.5, atol=Float64(1e-6))  # 5 m/s * 0.1


def test_hermite_prior_hits_endpoints() raises:
    var ts: List[Float64] = [0.0, 2.0, 4.0]
    var q = time_queries(ts, 0.0, 4.0)
    var p = hermite_prior(-2.0, 0.0, 0.0, 2.0, 1.0, 0.3, 4.0, 1.0, 0.5, 1.0, -0.5, q)
    assert_almost_equal(Float64(p.at2(0, 0)), -2.0, atol=Float64(1e-6))
    assert_almost_equal(Float64(p.at2(2, 0)), 2.0, atol=Float64(1e-6))
    assert_almost_equal(Float64(p.at2(2, 1)), 1.0, atol=Float64(1e-6))
    assert_almost_equal(Float64(p.at2(2, 2)), 0.3, atol=Float64(1e-6))
    var fr = completion_frame(_state(0.0, 0.0, 0.0, 1.0, 0.0), _state(1.0, 4.0, 4.0, 1.0, 0.0))
    assert_almost_equal(fr.ox, 2.0, atol=Float64(1e-12))
    assert_almost_equal(fr.yaw, pi / 4.0, atol=Float64(1e-12))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
