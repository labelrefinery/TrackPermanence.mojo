"""TrackPermanence CLI: Re-ID + completion plugin over tracker CSVs.

Usage:
    mojo run -I src src/main.mojo WEIGHTS.lft IN.csv OUT.csv [--threshold 0.5]

IN.csv holds tracker output in the labelrefinery CSV contract
(`track_id,cls,t,x,y,z,w,l,h,vx,vy,theta,conf`, global coordinates — e.g.
OfflinePoly.mojo's output); OUT.csv receives the merged, gap-completed tracks.
"""

from std.sys import argv

from trackpermanence.csvio import assign_frames, parse_f64, read_tracker_csv, write_tracker_csv
from trackpermanence.lft import load_lft
from trackpermanence.model import decode_config
from trackpermanence.plugin import PluginConfig, run_plugin
from trackpermanence.types import Tracklet


def main() raises:
    var args = argv()
    var paths = List[String]()
    var pc = PluginConfig()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--threshold":
            pc.threshold = parse_f64(String(args[i + 1]))
            i += 2
            continue
        paths.append(a)
        i += 1
    if len(paths) != 3:
        print("usage: mojo run -I src src/main.mojo WEIGHTS.lft IN.csv OUT.csv [--threshold 0.5]")
        return
    var w = load_lft(paths[0])
    var cfg = decode_config(w)
    var sources = List[List[Tracklet]]()
    sources.append(read_tracker_csv(paths[1]))
    var frame_times = assign_frames(sources)
    var n_in = 0
    for tr in sources[0]:
        n_in += tr.age()
    var res = run_plugin(w, cfg, pc, sources[0], frame_times)
    var n_out = 0
    for tr in res.tracks:
        n_out += tr.age()
    print(
        len(sources[0]), "tracklets in (", n_in, "states ) ->", res.n_links, "links ->",
        len(res.tracks), "tracks out (", n_out, "states )",
    )
    write_tracker_csv(paths[2], res.tracks)
    print("wrote", paths[2])
