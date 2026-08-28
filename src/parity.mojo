"""Parity check against PyTorch: runs the exported samples through the Mojo
forward passes and compares with the recorded outputs.

Usage: mojo run -I src src/parity.mojo data/weights.lft data
"""

from std.sys import argv

from trackpermanence.lft import load_lft
from trackpermanence.model import completion_poses, decode_config, reid_logit
from trackpermanence.tensor import Tensor


def _max_abs_diff(a: Tensor, b: Tensor) raises -> Float32:
    if a.numel() != b.numel():
        raise Error("parity: size mismatch")
    var m: Float32 = 0.0
    for i in range(a.numel()):
        var d = abs(a[i] - b[i])
        if d > m:
            m = d
    return m


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: parity WEIGHTS.lft SAMPLE_DIR")
        return
    var w = load_lft(String(args[1]))
    var cfg = decode_config(w)
    var dir = String(args[2])
    var worst: Float32 = 0.0
    for k in range(3):
        var rs = load_lft(dir + "/reid_sample_" + String(k) + ".lft")
        var logit = reid_logit(w, cfg, rs["hist"], rs["fut"])
        var d_reid = abs(logit - rs["logit"][0])
        var cs = load_lft(dir + "/completion_sample_" + String(k) + ".lft")
        var poses = completion_poses(w, cfg, cs["hist"], cs["fut"], cs["q"], cs["prior"])
        var d_comp = _max_abs_diff(poses, cs["out"])
        print(
            "sample", k, "reid logit", logit, "(torch", rs["logit"][0], ") diff", d_reid,
            "| completion", cs["q"].dim(0), "poses max diff", d_comp,
        )
        worst = max(worst, max(d_reid, d_comp))
    print("max abs diff vs PyTorch:", worst)
    if worst < 1e-4:
        print("PARITY: PASS")
    else:
        print("PARITY: FAIL")
