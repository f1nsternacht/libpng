import struct

import zlib
from zlib import crc32

from collections import deque
from itertools import cycle
from enum import IntEnum

class Chunk():
    @classmethod
    def from_buffer(cls, buf):
        """read a single chunk from the head of the buffer"""
        new = cls()
        length = struct.unpack(">I", buf[:4])[0]
        new.type = buf[4:8]
        new.body = buf[8:8+length]
        new.crc = struct.unpack(">I", buf[8+length:8+length+4])[0]
        new.valid = new.crc == crc32(new.type+new.body)
        return new
    
    @classmethod
    def create(cls, ctype, body):
        new = cls()
        new.type = ctype
        new.body = body
        new.crc = crc32(new.type+new.body)
        new.valid = True
        return new

    def total(self):
        """return length including type-, length- and checksum fields"""
        return len(self.body)+12

    def __repr__(self):
        return "Chunk[{}]:{}".format(str(self.type), self.total())

    def to_bytes(self, recalculate_crc=True):
        """turn Chunk back into bytes
        length and checksum are calculated appropriately"""
        buf  = struct.pack(">I", len(self.body))
        buf += self.type
        buf += self.body
        if recalculate_crc:
            buf += struct.pack(">I", crc32(self.type+self.body))
        else:
            buf += struct.pack(">I", self.crc)
        return buf

def create_comment(text, tag='Comment', compressed = False):
    body = b''
    body += tag.encode('ascii') + b'\x00'
    if compressed:
        ctype = b'zTXt'
        body += text.encode('ascii')
    else:
        ctype = b'tEXt'
        body += b'\x00' + zlib.compress(text.encode('ascii'), 9)
    return Chunk.create(ctype, body)

IEND = Chunk.create(b'IEND', b'')

CRITICAL_CHUNK_TYPES = [b'IHDR', b'PLTE', b'IDAT', b'IEND']

def parse_chunks(s):
    """parse bytes to list of Chunks"""
    i = 0
    chunks = []
    while i<len(s):
        try:
            c = Chunk.from_buffer(s[i:])
            if not c.valid:
                print(f"WARNING: chunk {i} checksum failed")
            if c.type[0]&(1<<5) == 0:
                if not c.type in CRITICAL_CHUNK_TYPES:
                    print(f"ERROR: chunk {i} of type {c.type} cannot be interpreted")
        except struct.error:
            print("WARNING: Orphan data after IEND")
            return chunks, s[i:]
        chunks.append(c)
        i += c.total()
        if c.type == b'IEND':
            if i<len(s):
                print("WARNING: Orphan data after IEND chunk")
                return chunks, s[i:]
    return chunks, None

def bits_to_bytes(x):
    return (x >> 3) + ((x & 7) > 0)

_bits_per_pixel = {}
_bits_per_pixel[0] = {1:1, 2:2, 4:4, 8:8, 16:16}
_bits_per_pixel[2] = {8:24, 16:48}
_bits_per_pixel[3] = {1:1, 2:2, 4:4, 8:8}
_bits_per_pixel[4] = {8:16, 16:32}
_bits_per_pixel[6] = {8:32, 16:64}

class ColorType(IntEnum):
    GRAYSCALE = 0
    RGB = 2
    PALETTE = 3
    GRAYSCALE_ALPHA = 4
    RGBA = 6
    

class IHDR():
    @classmethod
    def create(cls, width, height, bitdepth, colortype, compression=0, filtermethod=0, interlace=0):
        new = cls()
        new.width = width
        new.height = height
        new.bitdepth = bitdepth
        new.colortype = colortype
        new.compression = compression
        new.filtermethod = filtermethod
        new.interlace = interlace
        assert new.width>0, "ERROR: width must be positive"
        assert new.height>0, "ERROR: height must be positive"
        assert new.colortype in set([0, 2, 3, 4, 6]), "ERROR: {} is not a valid colortype".format(new.colortype)
        if new.colortype == 0:
            assert new.bitdepth in set([1,2,4,8,16]), "ERROR: {} is not a valid bitdepth for colortype {}".format(new.bitdepth, ColorType(new.colortype).name)
        elif new.colortype == 2:
            assert new.bitdepth in set([8,16]), "ERROR: {} is not a valid bitdepth for colortype {}".format(new.bitdepth, ColorType(new.colortype).name)
        elif new.colortype == 3:
            assert new.bitdepth in set([1,2,4,8]), "ERROR: {} is not a valid bitdepth for colortype {}".format(new.bitdepth, ColorType(new.colortype).name)
        elif new.colortype == 4:
            assert new.bitdepth in set([8,16]), "ERROR: {} is not a valid bitdepth for colortype {}".format(new.bitdepth, ColorType(new.colortype).name)
        elif new.colortype == 6:
            assert new.bitdepth in set([8,16]), "ERROR: {} is not a valid bitdepth for colortype {}".format(new.bitdepth, ColorType(new.colortype).name)
        assert new.compression==0, "ERROR: {} is not a valid compression type".format(new.compression)
        assert new.filtermethod==0, "ERROR: {} is not a valid filter method".format(new.filtermethod)
        assert new.interlace in set([0, 1]), "ERROR: {} is not a valid interlace method".format(new.interlace)
        new.bytes_per_pixel = bits_to_bytes(_bits_per_pixel[new.colortype][new.bitdepth])
        new.width_bytes = bits_to_bytes(new.width * _bits_per_pixel[new.colortype][new.bitdepth])
        return new

    @classmethod
    def from_buffer(cls, buf):
        length, expected = len(buf), struct.calcsize(">IIBBBBB")
        if length > expected:
            print(f"WARNING: IHDR too large? (found: {length}, expected: {expected})")
            raise ValueError
        elif length < expected:
            print(f"ERROR: IHDR too small? (found: {length}, expected: {expected})")
            raise ValueError
        width, height, bitdepth, colortype, compression, filtermethod, interlace = struct.unpack(">IIBBBBB", buf)
        new = cls.create(width, height, bitdepth, colortype, compression, filtermethod, interlace)
        return new

    def __repr__(self):
        data = [("width", self.width), ("height", self.height), ("bitdepth", self.bitdepth), ("colortype", ColorType(self.colortype).name), ("compression", self.compression), ("filtermethod", self.filtermethod), ("interlace", ["false", "adam7"][self.interlace])]
        return '\n'.join("{} : {}".format(y, x) for x, y in data)

    def to_chunk(self):
        c = Chunk()
        c.type = b'IHDR'
        c.body = struct.pack(">IIBBBBB", self.width, self.height, self.bitdepth, self.colortype, self.compression, self.filtermethod, self.interlace)
        return c


def group(seq, n):
    while seq:
        yield seq[:n]
        seq = seq[n:]

def split_to_chunks(data, size):
    """
    split encoded image data into IDAT chunks of given size
    """
    out = []
    for i in range(0, len(data), size):
        c = Chunk()
        c.type = b"IDAT"
        c.body = data[i:min(i + size, len(data))]
        out.append(c)
    return out

class FilterType(IntEnum):
    NONE = 0
    SUB = 1
    UP = 2
    AVG = 3
    PAETH = 4

def undo_sub(scanline, bytes_per_pixel):
    old = deque([0] * bytes_per_pixel)
    out = []
    for b in scanline:
        new = (b + old.pop()) & 0xff
        out.append(new)
        old.appendleft(new)
    return bytes(out)

def undo_up(scanline, prev, bytes_per_pixel):
    return bytes((x + b) & 0xff for x, b in zip(scanline, prev))

def undo_avg(scanline, prev, bytes_per_pixel):
    out = []
    old = deque([0] * bytes_per_pixel)
    for x, b in zip(scanline, prev):
        new = (x + (b + old.pop()) // 2) & 0xff
        out.append(new)
        old.appendleft(new)
    return bytes(out)

def undo_paeth(scanline, prev, bytes_per_pixel):
    out = []
    old = deque([0] * bytes_per_pixel)
    old_up = deque([0] * bytes_per_pixel)
    for x, b in zip(scanline, prev):
        a = old.pop()
        c = old_up.pop()
        p = a + b - c
        pa = abs(p - a)
        pb = abs(p - b)
        pc = abs(p - c)
        if pa <= pb and pa <= pc:
            rho = a
        elif pb <= pc:
            rho = b
        else:
            rho = c
        new = (x + rho) & 0xff
        out.append(new)
        old.appendleft(new)
        old_up.appendleft(b)
    return bytes(out)

def undo_filter(pixels, width, bytes_per_pixel):
    out = b''
    prev = bytes([0] * width)
    for row in group(pixels, width + 1):
        filter_type, *scanline = row
        if filter_type == FilterType.NONE:
            cur = bytes(scanline)
        elif filter_type == FilterType.SUB:
            cur = undo_sub(scanline, bytes_per_pixel)
        elif filter_type == FilterType.UP:
            cur = undo_up(scanline, prev, bytes_per_pixel)
        elif filter_type == FilterType.AVG:
            cur = undo_avg(scanline, prev, bytes_per_pixel)
        elif filter_type == FilterType.PAETH:
            cur = undo_paeth(scanline, prev, bytes_per_pixel)
        else:
            print(f"ERROR: Invalid filter type {filter_type}")
            raise ValueError
        assert len(cur) == width, f"ERROR: scanline wrong, is {len(cur)} should be {width}"
        out += cur
        prev = cur
    return bytearray(out)

def apply_sub(scanline, bytes_per_pixel):
    old = deque([0] * bytes_per_pixel)
    out = []
    for x in scanline:
        new = (x - old.pop()) & 0xff
        out.append(new)
        old.appendleft(new)
    return bytes(out)

def apply_up(scanline, prev, bytes_per_pixel):
    return bytes((x - b) & 0xff for x, b in zip(scanline, prev))

def apply_avg(scanline, prev, bytes_per_pixel):
    out = []
    old = deque([0] * bytes_per_pixel)
    for x, b in zip(scanline, prev):
        new = (x - (b + old.pop()) // 2) & 0xff
        out.append(new)
        old.appendleft(new)
    return bytes(out)

def apply_paeth(scanline, prev, bytes_per_pixel):
    out = []
    old = deque([0] * bytes_per_pixel)
    old_up = deque([0] * bytes_per_pixel)
    for x, b in zip(scanline, prev):
        a = old.pop()
        c = old.pop()
        p = a + b - c
        pa = abs(p - a)
        pb = abs(p - b)
        pc = abs(p - c)
        if pa <= pb and pa <= pc:
            rho = a
        elif pb <= pc:
            rho = b
        else:
            rho = c
        new = (x - rho) & 0xff
        out.append(new)
        old.appendleft(new)
        old_up.appendleft(b)
    return bytes(out)

def apply_filter(pixels, width, bytes_per_pixel, method):
    out = b''
    prev = bytes([0] * width)
    for filter_type, scanline in zip(method, group(pixels, width)):
        if filter_type == FilterType.NONE:
            cur = scanline
        elif filter_type == FilterType.SUB:
            cur = apply_sub(scanline, bytes_per_pixel)
        elif filter_type == FilterType.UP:
            cur = apply_up(scanline, prev, bytes_per_pixel)
        elif filter_type == FilterType.AVG:
            cur = apply_avg(scanline, prev, bytes_per_pixel)
        elif filter_type == FilterType.PAETH:
            cur = apply_paeth(scanline, prev, bytes_per_pixel)
        else:
            print(f"ERROR: Invalid filter type {filter_type}")
            raise ValueError
        row = b'{:d}{}'.format(filter_type, cur)
        out += row
        prev = row

def read_nth(seq, n, offset=0):
    for i in range(offset, len(seq), n):
        yield seq[i]

def merge_data(chunks):
    """concat content of all chunks of type IDAT"""
    return b''.join(chunk.body for chunk in chunks if chunk.type == b'IDAT')

def decompress(data):
    D = zlib.decompressobj()
    data = D.decompress(data)
    return data, D.unused_data

class PNG():
    MAGIC = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"
    
    @classmethod
    def from_chunks(cls, chunks, ihdr=None, orphan=None):
        new = cls()
        new.chunks = chunks
        if not orphan is None:
            new.orphan = orphan
        if ihdr is None:
            try:
                ihdr, *_ = new.get_chunks_by_type(b'IHDR')
            except ValueError:
                print("ERROR: Please supply a IHDR chunk")
            if len(_) > 0:
                print("WARNING: Multiple IHDR chunks found, used first")
            new.ihdr = IHDR.from_buffer(ihdr.body)
        else:
            new.ihdr = ihdr
            new.chunks = [ihdr.to_chunk()] + chunks
            if len(new.get_chunks_by_type(b'IHDR')) > 1:
                print("WARNING: Multiple IHDR chunks found, used the selected one")
        iend_chunks = new.get_chunks_by_type(b'IEND')
        if len(iend_chunks)==0:
            print("INFO: autogenerated IEND chunk")
            new.chunks.append(IEND)
        elif len(iend_chunks)>1:
            print(f"ERROR: there must be exactly one IEND chunk, but found {len(iend_chunks)}")
        elif len(iend_chunks[0].body) > 0:
            print(f"ERROR: IEND chunk must be empty, found {len(iend_chunks[0].body)} bytes")

        try:
            new.data, unused = decompress(merge_data(new.chunks))
            if unused:
                new.unused = unused
                print(f"INFO: unused {len(unused)} bytes of data in IDAT")
        except zlib.error:
            print("ERROR: zlib.decompress failed")
            return new
            
        if new.ihdr.interlace:
            print("ERROR: adam7 interlacing is currently not supported")
            return new

        actual_dim, supposed_dim = len(new.data), new.ihdr.width*new.ihdr.height*new.ihdr.bytes_per_pixel+new.ihdr.height
        if actual_dim>supposed_dim:
            print("WARNING: too many pixels")
            print("expected:\n\t{}x{}x{} = {}".format(new.ihdr.width, new.ihdr.height, new.ihdr.bytes_per_pixel, supposed_dim))
            print("found:\n\t{}".format(actual_dim))
        elif actual_dim<supposed_dim:
            print("ERROR: too few pixels")
            print("expected:\n\t{}x{}x{} = {}".format(new.ihdr.width, new.ihdr.height, new.ihdr.bytes_per_pixel, supposed_dim))
            print("found:\n\t{}".format(actual_dim))
            if False:
                raise ValueError
            else:
                new.data += (supposed_dim - actual_dim) * b'\x00'

        try:
            new.pixels = undo_filter(new.data, new.ihdr.width_bytes, new.ihdr.bytes_per_pixel)
        except AssertionError as e:
            print(e)
        except ValueError as e:
            pass
        pltes = new.get_chunks_by_type(b'PLTE')
        if new.ihdr.colortype != 3:
            if len(pltes) > 0:
                print(f"WARNING: there shouldn't be any PLTE chunks, {len(pltes)} found.")
        elif new.ihdr.colortype == 3:
            if len(pltes) > 1:
                print("WARNING: there should only be 1 PLTE chunk, {} found.".format(len(pltes)))
                return new
            new.plte, *_ = pltes
            new.pixels = [new.plte.body[x] for x in new.pixels]
        return new
        
    @classmethod
    def from_buffer(cls, buf, just_header=False):
        """interpret content of buffer as PNG image"""
        assert len(cls.MAGIC) + 12 < len(buf), "ERROR: buffer too short"
        if buf[:len(cls.MAGIC)] != cls.MAGIC:
            print("ERROR: magic bytes mismatched")
        
        orphan = None
        chunks, orphan = parse_chunks(buf[len(cls.MAGIC):])

        if chunks[0].type != b'IHDR':
            print("ERROR: first chunk must be of type IHDR, {} type chunk found".format(chunks[0].type))
        count = sum(chunk.type==b'IHDR' for chunk in chunks)
        if count > 1:
            print("ERROR: there must be exactly 1 IHDR chunk, found {}".format(count))
        if chunks[-1].type != b'IEND':
             print("ERROR: last chunk must be of type IEND, {} type chunk found".format(chunks[-1].type))
        count = sum(chunk.type==b'IEND' for chunk in chunks)
        if count!=1:
            print("ERROR: there must be exactly 1 IEND chunk, found {}".format(count))

        return cls.from_chunks(chunks, orphan = orphan)
        
    @classmethod
    def from_file(cls, name, just_header=False):
        """read file as a PNG image"""
        with open(name, "rb") as f:
            return cls.from_buffer(f.read(), just_header)

    def get_filter_bytes(self):
        if self.ihdr.interlace!=0:
            print("ERROR: can't deal with adam7 yet")
            raise NotImplemented
        return self.data[::self.ihdr.width*self.ihdr.bytes_per_pixel+1]

    def get_chunks_by_type(self, ctype):
        return [chunk for chunk in self.chunks if chunk.type == ctype]

    def rebuild_chunks(self, filtermethod=None, update_ihdr=True, compression_level=9):
        """
        build a new chunk list from pixel data.
        needs to be called, when pixel data was modified,
        before the image is saved.
        """
        if update_ihdr:
            chunks = [self.ihdr.to_chunk()]
        else:
            chunks = [self.chunks[0]]

        if filtermethod is None:
            if filtermethod is None:
                filtermethod = cycle((0,))
            self.data = apply_filter(self.pixels, self.ihdr.width_bytes, self.ihdr.bytes_per_pixel, filtermethod)
        for chunk in group(zlib.compress(data, compression_level), 8192):
            c = Chunk()
            c.type = b"IDAT"
            c.body = chunk
            chunks.append(c)

        chunks += [IEND]
        self.chunks = chunks

    def to_bytes(self, strip=False, compress=True, recalculate_crc=True):
        buf = self.MAGIC
        for c in self.chunks:
            buf += c.to_bytes(recalculate_crc)
        return buf

    def save(self, name, strip=False, compress=True, recalculate_crc=True):
        """
        write PNG to file
        changes made to the pixel data will only be saved if rebuild_chunks was called
        """
        with open(name, "wb") as f:
            return f.write(self.to_bytes(strip, compress, recalculate_crc))
    

if __name__=='__main__':
    import sys
    try:
        _, source = sys.argv
        img = PNG.from_file(source)
    except ValueError:
        pass
