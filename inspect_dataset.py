import os
import struct
import hashlib

def get_jpeg_size(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        if data[:2] != b"\xff\xd8":
            return None, None
        i = 2
        while i < len(data):
            if data[i] != 0xff:
                i += 1
                continue
            marker = data[i+1]
            if marker in (0xc0, 0xc1, 0xc2):
                h, w = struct.unpack(">HH", data[i+5:i+9])
                return w, h
            length = struct.unpack(">H", data[i+2:i+4])[0]
            i += 2 + length
    except Exception as e:
        return None, None
    return None, None

def get_hash(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

folder = r"C:\Users\shaik\OneDrive\Desktop\xray"
files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Total images found: {len(files)}")
print("-" * 50)
hashes = {}
duplicates = []

for f in sorted(files, key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 999):
    path = os.path.join(folder, f)
    w, h = get_jpeg_size(path)
    sz = os.path.getsize(path) / 1024
    file_hash = get_hash(path)
    
    dup_str = ""
    if file_hash in hashes:
        duplicates.append((f, hashes[file_hash]))
        dup_str = f" [DUPLICATE of {hashes[file_hash]}]"
    else:
        hashes[file_hash] = f
        
    print(f"{f:15} | Size: {sz:6.1f} KB | Resolution: {w}x{h}{dup_str}")

print("-" * 50)
print(f"Total unique images: {len(hashes)}")
print(f"Duplicates found: {len(duplicates)}")
if duplicates:
    for d1, d2 in duplicates:
        print(f"  - {d1} matches {d2}")
