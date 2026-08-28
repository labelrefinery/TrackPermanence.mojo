"""Re-ID and track completion forward passes on single un-padded samples,
mirroring TrackPermanence.py's `model.py` module by module. Weights are the
exported PyTorch state_dict keys prefixed with `reid.` / `completion.`."""

from .nn import (
    SeqState, broadcast_row, concat_cols, concat_rows, gru, layernorm, mha, mlp,
)
from .tensor import Tensor, add_, matmul_bias, relu_


struct ModelConfig(Copyable, Movable):
    var d_model: Int
    var nhead: Int

    def __init__(out self, d_model: Int, nhead: Int):
        self.d_model = d_model
        self.nhead = nhead


def decode_config(w: Dict[String, Tensor]) raises -> ModelConfig:
    ref c = w["__config__"]
    if c.numel() != 2:
        raise Error("bad __config__ tensor")
    return ModelConfig(Int(c[0]), Int(c[1]))


def _mlp(w: Dict[String, Tensor], p: String, x: Tensor) raises -> Tensor:
    return mlp(x, w[p + ".fc1.weight"], w[p + ".fc1.bias"], w[p + ".fc2.weight"], w[p + ".fc2.bias"])


def _gru(w: Dict[String, Tensor], p: String, x: Tensor, h0: Tensor, reverse: Bool) raises -> SeqState:
    return gru(
        x, w[p + ".cell.weight_ih"], w[p + ".cell.bias_ih"],
        w[p + ".cell.weight_hh"], w[p + ".cell.bias_hh"], h0, reverse,
    )


def _mha(w: Dict[String, Tensor], p: String, x_q: Tensor, x_kv: Tensor, nhead: Int) raises -> Tensor:
    return mha(
        x_q, x_kv, w[p + ".in_proj_weight"], w[p + ".in_proj_bias"],
        w[p + ".out_proj.weight"], w[p + ".out_proj.bias"], nhead,
    )


struct Encoding(Copyable, Movable):
    var h_hist: Tensor  # [1, d]
    var h_fut: Tensor  # [1, d]
    var out_hist: Tensor  # [Th, d]
    var out_fut: Tensor  # [Tf, d]

    def __init__(out self, var h_hist: Tensor, var h_fut: Tensor, var out_hist: Tensor, var out_fut: Tensor):
        self.h_hist = h_hist^
        self.h_fut = h_fut^
        self.out_hist = out_hist^
        self.out_fut = out_fut^


def motion_encoder(w: Dict[String, Tensor], p: String, hist: Tensor, fut: Tensor, d: Int) raises -> Encoding:
    """History GRU -> h_H; future U-GRU (forward from h_H, then backward
    over the forward outputs from its final state) -> h_F."""
    var zeros = Tensor([1, d])
    var eh = _mlp(w, p + ".hist_mlp", hist)
    var sh = _gru(w, p + ".hist_gru", eh, zeros, False)
    var ef = _mlp(w, p + ".fut_mlp", fut)
    var s1 = _gru(w, p + ".fut_fwd", ef, sh.h, False)
    var s2 = _gru(w, p + ".fut_bwd", s1.out, s1.h, True)
    return Encoding(sh.h.copy(), s2.h.copy(), sh.out.copy(), s2.out.copy())


def reid_logit(w: Dict[String, Tensor], cfg: ModelConfig, hist: Tensor, fut: Tensor) raises -> Float32:
    """Motion affinity logit for one (history [Th, 8], future [Tf, 8]) pair."""
    var enc = motion_encoder(w, "reid.enc", hist, fut, cfg.d_model)
    var x = concat_cols(enc.h_hist, enc.h_fut)
    var h = matmul_bias(x, w["reid.head1.weight"], w["reid.head1.bias"])
    relu_(h)
    var out = matmul_bias(h, w["reid.head2.weight"], w["reid.head2.bias"])
    return out[0]


def completion_poses(
    w: Dict[String, Tensor], cfg: ModelConfig, hist: Tensor, fut: Tensor, q: Tensor, prior: Tensor
) raises -> Tensor:
    """Refined occluded poses [Tq, 3] = (x, y, yaw) in the completion frame."""
    var p = String("completion")
    var enc = motion_encoder(w, p + ".enc", hist, fut, cfg.d_model)
    var mem = concat_rows(enc.out_hist, enc.out_fut)
    var qe = _mlp(w, p + ".q_mlp", q)
    var tq = q.dim(0)
    var a = _mha(w, p + ".cross", qe, mem, cfg.nhead)
    var cat = concat_cols(
        concat_cols(a, qe),
        concat_cols(broadcast_row(enc.h_hist, tq), broadcast_row(enc.h_fut, tq)),
    )
    var f0 = _mlp(w, p + ".fuse", cat)
    var p_init = matmul_bias(f0, w[p + ".init_head.weight"], w[p + ".init_head.bias"])
    add_(p_init, prior)
    var s = _mha(w, p + ".self_attn", f0, f0, cfg.nhead)
    add_(s, f0)
    var f1 = layernorm(s, w[p + ".norm.weight"], w[p + ".norm.bias"])
    var zeros = Tensor([1, cfg.d_model])
    var gf = _gru(w, p + ".bi_fwd", f1, zeros, False)
    var gb = _gru(w, p + ".bi_bwd", f1, zeros, True)
    var r = matmul_bias(concat_cols(gf.out, gb.out), w[p + ".ref1.weight"], w[p + ".ref1.bias"])
    relu_(r)
    var delta = matmul_bias(r, w[p + ".ref2.weight"], w[p + ".ref2.bias"])
    add_(p_init, delta)
    return p_init^
