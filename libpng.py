import struct

import zlib
from zlib import crc32

from collections import deque
from itertools import cycle

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
        if not new.valid:
            print("WARNING: crc failed")
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


def _parse_chunks(s):
    """parse bytes to list of Chunks"""
    i = 0
    chunks = []
    while i<len(s):
        try:
            c = Chunk.from_buffer(s[i:])
        except struct.error:
            print("WARNING: Orphan data after IEND")
            break
        chunks.append(c)
        i += c.total()
        if c.type==b'IEND':
            if i<len(s):
                print("WARNING: Orphan data after IEND chunk")
            break
    return chunks

def bits_to_bytes(x):
    return (x>>3) + ((x&7)>0)

_bits_per_pixel = {}
_bits_per_pixel[0] = {1:1, 2:2, 4:4, 8:8, 16:16}
_bits_per_pixel[2] = {8:24, 16:48}
_bits_per_pixel[3] = {1:1, 2:2, 4:4, 8:8}
_bits_per_pixel[4] = {8:16, 16:32}
_bits_per_pixel[6] = {8:32, 16:64}

class IHDR():
    @classmethod
    def from_buffer(cls, buf):
        new = cls()
        new.width, new.height, new.bitdepth, new.colortype, new.compression, new.filtermethod, new.interlace = struct.unpack(">IIBBBBB", buf)
        assert new.width>0, "ERROR: width must be positive"
        assert new.height>0, "ERROR: height must be positive"
        assert new.colortype in set([0, 2, 3, 4, 6]), "ERROR: {} is not a valid colortype".format(new.colortype)
        if new.colortype == 0:
            assert new.bitdepth in set([1,2,4,8,16]), "ERROR: {} is not a valid bitdepth for colortype {}".format(new.bitdepth, new.colortype)
        elif new.colortype == 2:
            assert new.bitdepth in set([8,16]), "ERROR: {} is not a valid bitdepth for colortype {}".format(new.bitdepth, new.colortype)
        elif new.colortype == 3:
            assert new.bitdepth in set([1,2,4,8]), "ERROR: {} is not a valid bitdepth for colortype {}".format(new.bitdepth, new.colortype)
        elif new.colortype == 4:
            assert new.bitdepth in set([8,16]), "ERROR: {} is not a valid bitdepth for colortype {}".format(new.bitdepth, new.colortype)
        elif new.colortype == 6:
            assert new.bitdepth in set([8,16]), "ERROR: {} is not a valid bitdepth for colortype {}".format(new.bitdepth, new.colortype)
        assert new.compression==0, "ERROR: {} is not a valid compression type".format(new.compression)
        assert new.filtermethod==0, "ERROR: {} is not a valid filter method".format(new.filtermethod)
        assert new.interlace in set([0, 1]), "ERROR: {} is not a valid interlace method".format(new.interlace)
        new.bytes_per_pixel = bits_to_bytes(_bits_per_pixel[new.colortype][new.bitdepth])
        new.width_bytes = bits_to_bytes(new.width*_bits_per_pixel[new.colortype][new.bitdepth])
        return new
    
    def __repr__(self):
        data = [("width", self.width), ("height", self.height), ("bitdepth", self.bitdepth), ("colortype", self.colortype), ("compression", self.compression), ("filtermethod", self.filtermethod), ("interlace", self.interlace)]
        return '\n'.join("{:d} : {:s}".format(y, x) for x, y in data)

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
    out = []
    for i in range(0, len(data), size):
        c = Chunk()
        c.type = b"IDAT"
        c.body = data[i:min(i+size, len(data))]
        out.append(c)
    return out

def restore_from_filter(pixels, width, bytes_per_pixel):
    def filter_sub(line):
        old = deque(0 for _ in range(bytes_per_pixel))
        out = []
        for b in line:
            new = (b+old.pop())&0xff
            out += [new]
            old.appendleft(new)
        out = bytes(out)
        return out
    
    def filter_up(line, prev):
        return b''.join(bytes(((a+b)&0xff,)) for a, b in zip(line, prev))
    
    def filter_avg(line, prev):
        out = []
        old = deque(0 for _ in range(bytes_per_pixel))
        for x, b in zip(line, prev):
            out += [(x+(b+old.pop())//2)&0xff]
            old.appendleft(out[-1])
        return bytes(out)
    
    def filter_paeth(line, prev):
        out = b''
        old = deque(0 for _ in range(bytes_per_pixel))
        old_up = deque(0 for _ in range(bytes_per_pixel))
        for x, b in zip(line, prev):
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
            out += bytes(((x + rho)&0xff,))
            old.appendleft(out[-1])
            old_up.appendleft(b)
        return out
    
    out = b""
    prev = b"\x00"*width
    for line in group(pixels, width+1):
        filter_type = line[0]
        if filter_type==0:
            cur = line[1:]
        elif filter_type==1:
            cur = filter_sub(line[1:])
        elif filter_type==2:
            cur = filter_up(line[1:], prev)
        elif filter_type==3:
            cur = filter_avg(line[1:], prev)
        elif filter_type==4:
            cur = filter_paeth(line[1:], prev)
        else:
            print("ERROR: Invalid filter type {}".format(filter_type))
            raise ValueError
        assert len(cur)==width, "missing some bytes"
        out += cur
        prev = cur
    return bytearray(out)

def apply_filter(pixels, width, bytes_per_pixel, method):
    def filter_sub(line):
        old = deque(0 for _ in range(bytes_per_pixel))
        out = []
        for x in line:
            new = (x - old.pop())%256
            out.append(new)
            old.appendleft(x)
        return bytes(out)

    def filter_up(line, prev):
        return bytes([(x-y)%256 for x, y in zip(line, prev)])

    def filter_avg(line, prev):
        out = []
        old = deque(0 for _ in range(bytes_per_pixel))
        for x, b in zip(line, prev):
            out += [(x-(b+old.pop())//2)&0xff]
            old.appendleft(x)
        return bytes(out)

    def filter_paeth(line, prev):
        out = []
        old = deque(0 for _ in range(bytes_per_pixel))
        old_up = deque(0 for _ in range(bytes_per_pixel))
        for x, b in zip(line, prev):
            a = old.pop()
            c = old_up.pop()
            p = a + b - c
            pa = abs(p - a)
            pb = abs(p - b)
            pc = abs(p - c)
            if pa<=pb and pa<=pc:
                rho = a
            elif pb<=pc:
                rho = b
            else:
                rho = c
            new = (x-rho)%256
            out.append(new)
            old.appendleft(x)
            old_up.appendleft(b)
        return bytes(out)

    out = b""
    prev = b"\x00"*width
    for switch, line in zip(method, group(pixels, width)):
        if switch==0:
            cur = b'\x00' + line
        elif switch==1:
            cur = b'\x01' + filter_sub(line)
        elif switch==2:
            cur = b'\x02' + filter_up(line, prev)
        elif switch==3:
            cur = b'\x03' + filter_avg(line, prev)
        elif switch==4:
            cur = b'\x04' + filter_paeth(line, prev)
        else:
            print("ERROR: invalid filter type {}".format(method))
            raise ValueError
        out += cur
        prev = line
    return out

def read_nth(seq, n, offset=0):
    for i in range(offset, len(seq), n):
        yield seq[i]

def _merge_data(chunks):
    """concat content of all chunks of type IDAT"""
    return b''.join(chunk.body for chunk in chunks if chunk.type==b'IDAT')

class PNG():
    MAGIC = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"
    
    @classmethod
    def from_buffer(cls, buf):
        """interpret content of buffer as PNG image"""
        new = cls()
        assert len(cls.MAGIC) + 12 < len(buf), "ERROR: buffer too short"
        assert buf[:len(cls.MAGIC)]==cls.MAGIC, "ERROR: magic bytes mismatched"
        
        new.chunks = _parse_chunks(buf[len(cls.MAGIC):])
        
        assert new.chunks[0].type == b'IHDR', "ERROR: first chunk must be of type IHDR, {} type chunk found".format(new.chunks[0].type)
        count = sum(chunk.type==b'IHDR' for chunk in new.chunks)
        assert count==1, "ERROR: there must be exactly 1 IHDR chunk, found {}".format(count)
        assert new.chunks[-1].type == b'IEND', "ERROR: last chunk must be of type IEND, {} type chunk found".format(new.chunks[-1].type)
        count = sum(chunk.type==b'IEND' for chunk in new.chunks)
        assert count==1, "ERROR: there must be exactly 1 IEND chunk, found {}".format(count)

        new.ihdr = IHDR.from_buffer(new.chunks[0].body)
        if new.ihdr.colortype==3:
            raise NotImplementedError("color type 3")
        if new.ihdr.interlace:
            raise NotImplementedError("adam7 interlacing")
        
        new.data = zlib.decompress(_merge_data(new.chunks))
        if new.ihdr.colortype==3:
            print("support for colour type 3 is not yet implemented")
            return new

        actual_dim, supposed_dim = len(new.data), new.ihdr.width*new.ihdr.height*new.ihdr.bytes_per_pixel+new.ihdr.height
        if actual_dim>supposed_dim:
            print("WARNING: too many pixels")
        elif actual_dim<supposed_dim:
            print("ERROR: too few pixels")
            raise ValueError

        try:
            new.pixels = restore_from_filter(new.data, new.ihdr.width_bytes, new.ihdr.bytes_per_pixel)
        except AssertionError as e:
            print(e)
        except ValueError as e:
            pass
        return new
    
    @classmethod
    def from_file(cls, name):
        """read file as a PNG image"""
        with open(name, "rb") as f:
            return cls.from_buffer(f.read())
    
    def rebuild_chunks(self, filtermethod=None, update_ihdr=True):
        """
        build a new chunk list from pixel data.
        needs to be called, when pixel data was modified, 
        before the image is saved.
        """
        if update_ihdr:
            chunks = [self.ihdr.to_chunk()]
        else:
            chunks = [self.chunks[0]]
        
        if filtermethod == None:
            filtermethod = cycle((0,))
        data = apply_filter(self.pixels, self.ihdr.width_bytes, self.ihdr.bytes_per_pixel, filtermethod) 
        for chunk in group(zlib.compress(data), 8192):
            c = Chunk()
            c.type = b"IDAT"
            c.body = chunk
            chunks.append(c)
        
        chunks += [self.chunks[-1]]
        self.chunks = chunks
    
    def save(self, name, strip=False, compress=True, recalculate_crc=True):
        """
        write PNG to file
        changes made to the pixel data will only be saved if rebuild_chunks was called
        """
        buf = self.MAGIC
        for c in self.chunks:
            buf += c.to_bytes(recalculate_crc)
        with open(name, "wb") as f:
            return f.write(buf)
    

if __name__=='__main__':
    import sys
    _, source, target = sys.argv
    img = PNG.from_file(source)
    img.rebuild_chunks()
    img.save(target)
