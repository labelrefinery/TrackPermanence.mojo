"""Neural-net primitives matching the PyTorch modules used by TrackPermanence:
2-layer ReLU MLP, GRUCell recurrence (forward or reverse), single-sequence
multi-head attention (nn.MultiheadAttention, batch_first, no masks) and
LayerNorm. All tensors are row-major [T, features]."""

from std.math import exp, sqrt, tanh

from .tensor import Tensor, add_, matmul_bias, relu_


def sigmoid(x: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-x))


def row(x: Tensor, i: Int) -> Tensor:
    """Row `i` of a [T, k] tensor as a [1, k] tensor."""
    var k = x.dim(1)
    var out = Tensor([1, k])
    for c in range(k):
        out[c] = x.at2(i, c)
    return out^


def set_row(mut x: Tensor, i: Int, r: Tensor):
    for c in range(x.dim(1)):
        x.set2(i, c, r[c])


def slice_rows(w: Tensor, start: Int, n: Int) -> Tensor:
    """Rows [start, start + n) of a 2-D tensor (or elements of a 1-D one)."""
    if w.rank() == 1:
        var out = Tensor([n])
        for i in range(n):
            out[i] = w[start + i]
        return out^
    var k = w.dim(1)
    var out = Tensor([n, k])
    for i in range(n):
        for c in range(k):
            out.set2(i, c, w.at2(start + i, c))
    return out^


def concat_cols(a: Tensor, b: Tensor) raises -> Tensor:
    if a.dim(0) != b.dim(0):
        raise Error("concat_cols: row mismatch")
    var out = Tensor([a.dim(0), a.dim(1) + b.dim(1)])
    for i in range(a.dim(0)):
        for c in range(a.dim(1)):
            out.set2(i, c, a.at2(i, c))
        for c in range(b.dim(1)):
            out.set2(i, a.dim(1) + c, b.at2(i, c))
    return out^


def concat_rows(a: Tensor, b: Tensor) raises -> Tensor:
    if a.dim(1) != b.dim(1):
        raise Error("concat_rows: column mismatch")
    var out = Tensor([a.dim(0) + b.dim(0), a.dim(1)])
    for i in range(a.dim(0)):
        for c in range(a.dim(1)):
            out.set2(i, c, a.at2(i, c))
    for i in range(b.dim(0)):
        for c in range(a.dim(1)):
            out.set2(a.dim(0) + i, c, b.at2(i, c))
    return out^


def broadcast_row(h: Tensor, t: Int) -> Tensor:
    var k = h.numel()
    var out = Tensor([t, k])
    for i in range(t):
        for c in range(k):
            out.set2(i, c, h[c])
    return out^


def mlp(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) raises -> Tensor:
    """relu(fc2(relu(fc1(x))))."""
    var h = matmul_bias(x, w1, b1)
    relu_(h)
    var out = matmul_bias(h, w2, b2)
    relu_(out)
    return out^


struct SeqState(Copyable, Movable):
    """Outputs of a recurrence: per-step states [T, d] and the final state [1, d]."""

    var out: Tensor
    var h: Tensor

    def __init__(out self, var out_seq: Tensor, var h: Tensor):
        self.out = out_seq^
        self.h = h^


def gru(
    x: Tensor,
    w_ih: Tensor,
    b_ih: Tensor,
    w_hh: Tensor,
    b_hh: Tensor,
    h0: Tensor,
    reverse: Bool,
) raises -> SeqState:
    """torch.nn.GRUCell unrolled over the rows of `x` ([T, in]); gate order
    r, z, n in the stacked [3d, *] weights. `h0` is [1, d] (zeros allowed).
    `reverse` traverses the rows last-to-first (outputs stay in row order)."""
    var t = x.dim(0)
    var d = w_hh.dim(1)
    var h = h0.copy()
    var out = Tensor([t, d])
    for step in range(t):
        var i = t - 1 - step if reverse else step
        var gi = matmul_bias(row(x, i), w_ih, b_ih)  # [1, 3d]
        var gh = matmul_bias(h, w_hh, b_hh)  # [1, 3d]
        var h_new = Tensor([1, d])
        for c in range(d):
            var r = sigmoid(gi[c] + gh[c])
            var z = sigmoid(gi[d + c] + gh[d + c])
            var n = tanh(gi[2 * d + c] + r * gh[2 * d + c])
            h_new[c] = (1.0 - z) * n + z * h[c]
        h = h_new^
        set_row(out, i, h)
    return SeqState(out^, h^)


def softmax_rows_(mut s: Tensor):
    var n = s.dim(0)
    var m = s.dim(1)
    for i in range(n):
        var mx = s.at2(i, 0)
        for j in range(1, m):
            if s.at2(i, j) > mx:
                mx = s.at2(i, j)
        var total: Float32 = 0.0
        for j in range(m):
            var e = exp(s.at2(i, j) - mx)
            s.set2(i, j, e)
            total += e
        for j in range(m):
            s.set2(i, j, s.at2(i, j) / total)


def mha(
    x_q: Tensor,
    x_kv: Tensor,
    in_w: Tensor,
    in_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    nhead: Int,
) raises -> Tensor:
    """nn.MultiheadAttention(d, nhead, batch_first=True) on one un-padded
    sequence pair: x_q [Tq, d], x_kv [Tk, d] -> [Tq, d]."""
    var d = x_q.dim(1)
    if d % nhead != 0:
        raise Error("mha: d_model not divisible by nhead")
    var dh = d // nhead
    var q = matmul_bias(x_q, slice_rows(in_w, 0, d), slice_rows(in_b, 0, d))
    var k = matmul_bias(x_kv, slice_rows(in_w, d, d), slice_rows(in_b, d, d))
    var v = matmul_bias(x_kv, slice_rows(in_w, 2 * d, d), slice_rows(in_b, 2 * d, d))
    var tq = x_q.dim(0)
    var tk = x_kv.dim(0)
    var scale = Float32(1.0) / sqrt(Float32(dh))
    var ctx = Tensor([tq, d])
    for h in range(nhead):
        var scores = Tensor([tq, tk])
        for i in range(tq):
            for j in range(tk):
                var acc: Float32 = 0.0
                for c in range(dh):
                    acc += q.at2(i, h * dh + c) * k.at2(j, h * dh + c)
                scores.set2(i, j, acc * scale)
        softmax_rows_(scores)
        for i in range(tq):
            for c in range(dh):
                var acc: Float32 = 0.0
                for j in range(tk):
                    acc += scores.at2(i, j) * v.at2(j, h * dh + c)
                ctx.set2(i, h * dh + c, acc)
    return matmul_bias(ctx, out_w, out_b)


def layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: Float32 = 1e-5) -> Tensor:
    var t = x.dim(0)
    var d = x.dim(1)
    var out = Tensor([t, d])
    for i in range(t):
        var mean: Float32 = 0.0
        for c in range(d):
            mean += x.at2(i, c)
        mean /= Float32(d)
        var var_: Float32 = 0.0
        for c in range(d):
            var diff = x.at2(i, c) - mean
            var_ += diff * diff
        var_ /= Float32(d)
        var inv = Float32(1.0) / sqrt(var_ + eps)
        for c in range(d):
            out.set2(i, c, (x.at2(i, c) - mean) * inv * gamma[c] + beta[c])
    return out^
