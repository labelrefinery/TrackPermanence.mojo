"""Offline Re-ID + track completion over tracker CSVs (mirrors
TrackPermanence.py's `infer.py`): link terminated tracklets to later ones by
Re-ID score (greedy, chains allowed), fill each gap with the completion net
at the scene frame rate, emit merged tracks."""

from std.math import min, max

from .features import (
    LocalFrame, completion_frame, hermite_prior, motion_features, time_queries, wrap_angle,
)
from .model import ModelConfig, completion_poses, reid_logit
from .nn import sigmoid
from .tensor import Tensor
from .types import BoxState, Tracklet


struct PluginConfig(Copyable, Movable):
    var max_hist_s: Float64
    var min_gap_s: Float64
    var max_gap_s: Float64
    var max_future_s: Float64
    var threshold: Float64

    def __init__(out self):
        self.max_hist_s = 2.5
        self.min_gap_s = 1.5
        self.max_gap_s = 12.5
        self.max_future_s = 2.5
        self.threshold = 0.5


@fieldwise_init
struct Link(ImplicitlyCopyable, Movable):
    var hist: Int
    var fut: Int
    var score: Float64


def tail(states: List[BoxState], seconds: Float64) -> List[BoxState]:
    """Last `seconds` of a state list (t > t_end - seconds, inclusive start)."""
    var t_end = states[len(states) - 1].t
    var out = List[BoxState]()
    for s in states:
        if s.t >= t_end - seconds:
            out.append(s)
    return out^


def head(states: List[BoxState], seconds: Float64) -> List[BoxState]:
    var t_start = states[0].t
    var out = List[BoxState]()
    for s in states:
        if s.t <= t_start + seconds:
            out.append(s)
    return out^


def reid_score(
    w: Dict[String, Tensor], cfg: ModelConfig, pc: PluginConfig,
    hist: List[BoxState], fut: List[BoxState],
) raises -> Float64:
    var last = hist[len(hist) - 1]
    var frame = LocalFrame(last.x, last.y, last.theta)
    var h = motion_features(tail(hist, pc.max_hist_s), frame, last.t)
    var f = motion_features(head(fut, pc.max_future_s), frame, last.t)
    return Float64(sigmoid(reid_logit(w, cfg, h, f)))


def link_tracklets(
    w: Dict[String, Tensor], cfg: ModelConfig, pc: PluginConfig, tracklets: List[Tracklet]
) raises -> List[Link]:
    var n = len(tracklets)
    var scene_end = 0.0
    for tr in tracklets:
        scene_end = max(scene_end, tr.states[len(tr.states) - 1].t)
    var cands = List[Link]()
    for i in range(n):
        var end_i = tracklets[i].states[len(tracklets[i].states) - 1].t
        if scene_end - end_i < pc.min_gap_s:
            continue
        for j in range(n):
            if j == i or tracklets[j].cls != tracklets[i].cls:
                continue
            var gap = tracklets[j].states[0].t - end_i
            if gap < pc.min_gap_s or gap > pc.max_gap_s:
                continue
            var s = reid_score(w, cfg, pc, tracklets[i].states, tracklets[j].states)
            if s >= pc.threshold:
                cands.append(Link(i, j, s))
    # sort by descending score (insertion sort; candidate lists are small)
    for a in range(1, len(cands)):
        var key = cands[a]
        var b = a - 1
        while b >= 0 and cands[b].score < key.score:
            cands[b + 1] = cands[b]
            b -= 1
        cands[b + 1] = key
    var used_h = List[Bool](length=n, fill=False)
    var used_f = List[Bool](length=n, fill=False)
    var links = List[Link]()
    for c in cands:
        if used_h[c.hist] or used_f[c.fut]:
            continue
        used_h[c.hist] = True
        used_f[c.fut] = True
        links.append(c)
    return links^


def complete_gap(
    w: Dict[String, Tensor], cfg: ModelConfig, pc: PluginConfig,
    hist: List[BoxState], fut: List[BoxState], frame_times: List[Float64], tid: Int,
) raises -> List[BoxState]:
    """Poses for every scene frame strictly inside the gap between the last
    history state and the first future state."""
    var last = hist[len(hist) - 1]
    var first = fut[0]
    var t0 = last.t
    var t1 = first.t
    var t_miss = List[Float64]()
    for t in frame_times:
        if t > t0 + 1e-9 and t < t1 - 1e-9:
            t_miss.append(t)
    var out = List[BoxState]()
    if len(t_miss) == 0:
        return out^
    var frame = completion_frame(last, first)
    var h = motion_features(tail(hist, pc.max_hist_s), frame, t0)
    var f = motion_features(head(fut, pc.max_future_s), frame, t0)
    var q = time_queries(t_miss, t0, t1 - t0)
    var p0 = frame.to_local(last.x, last.y)
    var p1 = frame.to_local(first.x, first.y)
    var v0 = frame.vec_to_local(last.vx, last.vy)
    var v1 = frame.vec_to_local(first.vx, first.vy)
    var prior = hermite_prior(
        Float32(p0[0]), Float32(p0[1]), Float32(frame.yaw_to_local(last.theta)),
        Float32(p1[0]), Float32(p1[1]), Float32(frame.yaw_to_local(first.theta)),
        Float32(t1 - t0), Float32(v0[0]), Float32(v0[1]), Float32(v1[0]), Float32(v1[1]), q,
    )
    var poses = completion_poses(w, cfg, h, f, q, prior)
    for k in range(len(t_miss)):
        var a = (t_miss[k] - t0) / (t1 - t0)
        var g = frame.to_global(Float64(poses.at2(k, 0)), Float64(poses.at2(k, 1)))
        out.append(BoxState(
            frame=-1, t=t_miss[k], x=g[0], y=g[1],
            z=last.z + a * (first.z - last.z),
            w=0.5 * (last.w + first.w), l=0.5 * (last.l + first.l), h=0.5 * (last.h + first.h),
            vx=0.0, vy=0.0, theta=frame.yaw_to_global(Float64(poses.at2(k, 2))),
            conf=min(last.conf, first.conf), cls=last.cls, tid=tid,
        ))
    # velocities from the completed positions (central differences)
    for k in range(len(out)):
        var prev = last if k == 0 else out[k - 1]
        var nxt = first if k == len(out) - 1 else out[k + 1]
        out[k].vx = (nxt.x - prev.x) / (nxt.t - prev.t)
        out[k].vy = (nxt.y - prev.y) / (nxt.t - prev.t)
    return out^


struct PluginResult(Copyable, Movable):
    var tracks: List[Tracklet]
    var n_links: Int

    def __init__(out self, var tracks: List[Tracklet], n_links: Int):
        self.tracks = tracks^
        self.n_links = n_links


def run_plugin(
    w: Dict[String, Tensor], cfg: ModelConfig, pc: PluginConfig,
    tracklets: List[Tracklet], frame_times: List[Float64],
) raises -> PluginResult:
    var links = link_tracklets(w, cfg, pc, tracklets)
    var n = len(tracklets)
    var nxt = List[Int](length=n, fill=-1)
    var is_fut = List[Bool](length=n, fill=False)
    for l in links:
        nxt[l.hist] = l.fut
        is_fut[l.fut] = True
    var out = List[Tracklet]()
    for i in range(n):
        if is_fut[i]:
            continue
        var merged = Tracklet(tracklets[i].tid, tracklets[i].cls)
        for s in tracklets[i].states:
            merged.states.append(s)
        var cur = i
        while nxt[cur] >= 0:
            var j = nxt[cur]
            for s in complete_gap(w, cfg, pc, merged.states, tracklets[j].states, frame_times, merged.tid):
                merged.states.append(s)
            for s in tracklets[j].states:
                var t = s
                t.tid = merged.tid
                merged.states.append(t)
            cur = j
        out.append(merged^)
    return PluginResult(out^, len(links))
