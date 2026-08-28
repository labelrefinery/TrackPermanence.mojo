"""Local-frame feature construction, mirroring TrackPermanence.py's
`features.py` (same scaling constants and frame conventions)."""

from std.math import atan2, cos, floor, pi, sin, sqrt

from .tensor import Tensor
from .types import BoxState

comptime POS_SCALE = 0.1
comptime TIME_SCALE = 0.2
comptime VEL_SCALE = 0.1


def wrap_angle(a: Float64) -> Float64:
    var x = a
    while x > pi:
        x -= 2.0 * pi
    while x < -pi:
        x += 2.0 * pi
    return x


struct LocalFrame(Copyable, Movable):
    """Rigid BEV frame: origin + yaw; `to_local` maps global -> local."""

    var ox: Float64
    var oy: Float64
    var yaw: Float64

    def __init__(out self, ox: Float64, oy: Float64, yaw: Float64):
        self.ox = ox
        self.oy = oy
        self.yaw = yaw

    def to_local(self, x: Float64, y: Float64) -> SIMD[DType.float64, 2]:
        var c = cos(self.yaw)
        var s = sin(self.yaw)
        var dx = x - self.ox
        var dy = y - self.oy
        return SIMD[DType.float64, 2](c * dx + s * dy, -s * dx + c * dy)

    def to_global(self, x: Float64, y: Float64) -> SIMD[DType.float64, 2]:
        var c = cos(self.yaw)
        var s = sin(self.yaw)
        return SIMD[DType.float64, 2](c * x - s * y + self.ox, s * x + c * y + self.oy)

    def vec_to_local(self, vx: Float64, vy: Float64) -> SIMD[DType.float64, 2]:
        var c = cos(self.yaw)
        var s = sin(self.yaw)
        return SIMD[DType.float64, 2](c * vx + s * vy, -s * vx + c * vy)

    def yaw_to_local(self, yaw: Float64) -> Float64:
        return wrap_angle(yaw - self.yaw)

    def yaw_to_global(self, yaw: Float64) -> Float64:
        return wrap_angle(yaw + self.yaw)


def motion_features(states: List[BoxState], frame: LocalFrame, t0: Float64) -> Tensor:
    """[T, 8] = [x, y, yaw, t, cos, sin, vx, vy] in `frame`, scaled; shared by
    Re-ID and completion."""
    var out = Tensor([len(states), 8])
    for i in range(len(states)):
        var s = states[i]
        var p = frame.to_local(s.x, s.y)
        var th = frame.yaw_to_local(s.theta)
        var v = frame.vec_to_local(s.vx, s.vy)
        out.set2(i, 0, Float32(p[0] * POS_SCALE))
        out.set2(i, 1, Float32(p[1] * POS_SCALE))
        out.set2(i, 2, Float32(th))
        out.set2(i, 3, Float32((s.t - t0) * TIME_SCALE))
        out.set2(i, 4, Float32(cos(th)))
        out.set2(i, 5, Float32(sin(th)))
        out.set2(i, 6, Float32(v[0] * VEL_SCALE))
        out.set2(i, 7, Float32(v[1] * VEL_SCALE))
    return out^


def time_queries(t_missing: List[Float64], t0: Float64, t_gap: Float64) -> Tensor:
    var out = Tensor([len(t_missing), 2])
    for i in range(len(t_missing)):
        var rel = t_missing[i] - t0
        out.set2(i, 0, Float32(rel * TIME_SCALE))
        out.set2(i, 1, Float32(rel / t_gap))
    return out^


def completion_frame(last: BoxState, first: BoxState, min_dist: Float64 = 0.5) -> LocalFrame:
    """Origin at the gap midpoint, x-axis along the endpoint-to-endpoint
    direction (last history heading for near-stationary gaps)."""
    var dx = first.x - last.x
    var dy = first.y - last.y
    var yaw = last.theta
    if sqrt(dx * dx + dy * dy) >= min_dist:
        yaw = atan2(dy, dx)
    return LocalFrame((last.x + first.x) * 0.5, (last.y + first.y) * 0.5, yaw)


def hermite_prior(
    x0: Float32, y0: Float32, yaw0: Float32, x1: Float32, y1: Float32, yaw1: Float32,
    t_gap: Float32, vx0: Float32, vy0: Float32, vx1: Float32, vy1: Float32, q: Tensor,
) -> Tensor:
    """Cubic Hermite interpolation of (x, y) from endpoint poses+velocities,
    linear yaw, from q[:, 1] = t / t_gap; float32 like the torch version."""
    var tq = q.dim(0)
    var out = Tensor([tq, 3])
    # torch.remainder(yaw1 - yaw0 + pi, 2*pi) - pi
    var two_pi = Float32(2.0 * pi)
    var raw = yaw1 - yaw0 + Float32(pi)
    var dyaw = raw - two_pi * floor(raw / two_pi) - Float32(pi)
    for i in range(tq):
        var a = q.at2(i, 1)
        var a2 = a * a
        var a3 = a2 * a
        var h00 = 2.0 * a3 - 3.0 * a2 + 1.0
        var h10 = a3 - 2.0 * a2 + a
        var h01 = -2.0 * a3 + 3.0 * a2
        var h11 = a3 - a2
        out.set2(i, 0, h00 * x0 + h10 * vx0 * t_gap + h01 * x1 + h11 * vx1 * t_gap)
        out.set2(i, 1, h00 * y0 + h10 * vy0 * t_gap + h01 * y1 + h11 * vy1 * t_gap)
        out.set2(i, 2, yaw0 + a * dyaw)
    return out^
