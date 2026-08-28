"""Reader for the LFT1 tensor container written by TrackPermanence.py's
scripts/export_mojo.py (same format as LabelFormer.mojo).

Layout (little-endian):
    b"LFT1" | u32 n_tensors | per tensor:
        u32 name_len | name utf8 | u32 ndim | u32 shape[ndim] | f32 data (C order)
"""

from std.memory import bitcast

from .tensor import Tensor


def _u32(bytes: List[UInt8], off: Int) -> UInt32:
    var v: UInt32 = 0
    v |= UInt32(bytes[off])
    v |= UInt32(bytes[off + 1]) << 8
    v |= UInt32(bytes[off + 2]) << 16
    v |= UInt32(bytes[off + 3]) << 24
    return v


def _f32(bytes: List[UInt8], off: Int) -> Float32:
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](_u32(bytes, off)))[0]


def load_lft(path: String) raises -> Dict[String, Tensor]:
    """Parse an LFT1 file into a name -> Tensor dictionary."""
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    if len(bytes) < 8 or bytes[0] != 76 or bytes[1] != 70 or bytes[2] != 84 or bytes[3] != 49:
        raise Error("not an LFT1 file: " + path)
    var out = Dict[String, Tensor]()
    var n_tensors = Int(_u32(bytes, 4))
    var off = 8
    for _ in range(n_tensors):
        var name_len = Int(_u32(bytes, off))
        off += 4
        var name = String("")
        for i in range(name_len):
            name += chr(Int(bytes[off + i]))
        off += name_len
        var ndim = Int(_u32(bytes, off))
        off += 4
        var shape = List[Int]()
        var numel = 1
        for _ in range(ndim):
            var d = Int(_u32(bytes, off))
            off += 4
            shape.append(d)
            numel *= d
        if ndim == 0:
            shape.append(1)
        var t = Tensor(shape)
        for i in range(numel):
            t[i] = _f32(bytes, off)
            off += 4
        out[name] = t^
    if off != len(bytes):
        raise Error("trailing bytes in " + path)
    return out^
