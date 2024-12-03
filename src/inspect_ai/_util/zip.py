import os
import struct
import zlib
from typing import Any, BinaryIO, List


class ZipAppender:
    ZIP_HEADER: bytes = b"PK\x03\x04"
    CENTRAL_DIR_HEADER: bytes = b"PK\x01\x02"
    END_CENTRAL_DIR: bytes = b"PK\x05\x06"

    def __init__(self, file: BinaryIO) -> None:
        self.file: BinaryIO = file
        self.existing_entries: List[bytes] = []
        self._load_central_dir()

    def _load_central_dir(self) -> None:
        eocd_pos: int = self._find_end_central_dir()
        self.file.seek(eocd_pos)

        eocd: bytes = self.file.read(22)
        if not eocd.startswith(self.END_CENTRAL_DIR):
            raise ValueError("Invalid end of central directory record")

        num_entries: int = struct.unpack("<H", eocd[8:10])[0]
        struct.unpack("<L", eocd[12:16])[0]  # cd_size (unused)
        cd_offset: int = struct.unpack("<L", eocd[16:20])[0]

        self.file.seek(cd_offset)
        for _ in range(num_entries):
            entry_data: bytes = self._read_central_dir_entry()
            self.existing_entries.append(entry_data)

    def _find_end_central_dir(self) -> int:
        file_size: int = self.file.seek(0, os.SEEK_END)
        chunk_size: int = min(1024, file_size)
        self.file.seek(-chunk_size, os.SEEK_END)
        data: bytes = self.file.read()
        pos: int = data.rfind(self.END_CENTRAL_DIR)
        if pos == -1:
            raise ValueError("Could not find end of central directory")
        return file_size - chunk_size + pos

    def _read_central_dir_entry(self) -> bytes:
        entry: bytes = self.file.read(46)
        name_length: int = struct.unpack("<H", entry[28:30])[0]
        extra_length: int = struct.unpack("<H", entry[30:32])[0]
        comment_length: int = struct.unpack("<H", entry[32:34])[0]

        filename: bytes = self.file.read(name_length)
        extra: bytes = self.file.read(extra_length)
        comment: bytes = self.file.read(comment_length)

        return entry + filename + extra + comment

    def append_file(self, filename: str, data: bytes) -> None:
        local_header_offset: int = self.file.tell()
        compressor = zlib.compressobj(level=9, wbits=-15)
        compressed_data: bytes = compressor.compress(data) + compressor.flush()
        crc: int = zlib.crc32(data)

        # Set general purpose bit flag to indicate UTF-8 encoding (bit 11)
        general_purpose_flag: int = 0x0800

        encoded_filename: bytes = filename.encode("utf-8")
        header: bytes = (
            self.ZIP_HEADER
            + struct.pack("<H", 20)  # Version needed
            + struct.pack("<H", general_purpose_flag)  # Flags with UTF-8 encoding
            + struct.pack("<H", 8)  # Compression method
            + struct.pack("<H", 0)  # Time
            + struct.pack("<H", 0)  # Date
            + struct.pack("<L", crc)
            + struct.pack("<L", len(compressed_data))
            + struct.pack("<L", len(data))
            + struct.pack("<H", len(encoded_filename))
            + struct.pack("<H", 0)
            + encoded_filename
        )

        self.file.write(header)
        self.file.write(compressed_data)

        cd_entry: bytes = (
            self.CENTRAL_DIR_HEADER
            + struct.pack("<H", 20)  # Version made by
            + struct.pack("<H", 20)  # Version needed
            + struct.pack("<H", general_purpose_flag)  # Flags with UTF-8 encoding
            + struct.pack("<H", 8)  # Compression method
            + struct.pack("<H", 0)  # Time
            + struct.pack("<H", 0)  # Date
            + struct.pack("<L", crc)
            + struct.pack("<L", len(compressed_data))
            + struct.pack("<L", len(data))
            + struct.pack("<H", len(encoded_filename))
            + struct.pack("<H", 0)
            + struct.pack("<H", 0)
            + struct.pack("<H", 0)
            + struct.pack("<H", 0)
            + struct.pack("<L", 0)
            + struct.pack("<L", local_header_offset)
            + encoded_filename
        )

        self.existing_entries.append(cd_entry)

        cd_offset: int = self.file.tell()
        for entry in self.existing_entries:
            self.file.write(entry)

        eocd: bytes = (
            self.END_CENTRAL_DIR
            + struct.pack("<H", 0)  # Disk number
            + struct.pack("<H", 0)  # Disk with central directory
            + struct.pack("<H", len(self.existing_entries))  # Number of entries on disk
            + struct.pack("<H", len(self.existing_entries))  # Total number of entries
            + struct.pack(
                "<L", self.file.tell() - cd_offset
            )  # Size of central directory
            + struct.pack("<L", cd_offset)  # Offset of central directory
            + struct.pack("<H", 0)  # Comment length
        )
        self.file.write(eocd)

    def __enter__(self) -> "ZipAppender":
        return self

    def __exit__(self, *execinfo: Any) -> None:
        self.file.flush()
        self.file.close()
