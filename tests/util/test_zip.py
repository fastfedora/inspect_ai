import contextlib
import io
import tempfile
import zipfile
from typing import BinaryIO, Iterator, List, Set, Tuple
from zipfile import ZipFile

import pytest

from inspect_ai._util.file import file
from inspect_ai._util.zip import ZipAppender


@pytest.fixture
def sample_zip() -> io.BytesIO:
    """Create a temporary ZIP file with one entry for testing."""
    buffer: io.BytesIO = io.BytesIO()
    create_zip(buffer)
    return buffer


def create_zip(buffer: BinaryIO) -> None:
    with ZipFile(buffer, "w") as zf:
        zf.writestr("initial.txt", b"initial content")
    buffer.seek(0)


@contextlib.contextmanager
def temporary_zip_file() -> Iterator[BinaryIO]:
    with tempfile.NamedTemporaryFile() as temp:
        with file(temp.name, "wb+") as f:
            create_zip(f)
            yield f


def test_read_existing_zip(sample_zip: BinaryIO) -> None:
    """Test that we can correctly read an existing ZIP file."""
    appender: ZipAppender = ZipAppender(sample_zip)
    assert len(appender.existing_entries) == 1


def test_read_existing_zip_file() -> None:
    with temporary_zip_file() as f:
        test_read_existing_zip(f)


def test_append_single(sample_zip: BinaryIO) -> None:
    """Test appending a single new file to an existing ZIP."""
    appender: ZipAppender = ZipAppender(sample_zip)
    appender.append_file("new.txt", b"new content")

    sample_zip.seek(0)
    with ZipFile(sample_zip) as zf:
        namelist: Set[str] = set(zf.namelist())
        assert namelist == {"initial.txt", "new.txt"}
        assert zf.read("initial.txt") == b"initial content"
        assert zf.read("new.txt") == b"new content"


def test_append_single_file() -> None:
    with temporary_zip_file() as f:
        test_append_single(f)


def test_append_multiple(sample_zip: BinaryIO) -> None:
    """Test appending multiple files sequentially."""
    appender: ZipAppender = ZipAppender(sample_zip)

    files_to_add: List[Tuple[str, bytes]] = [
        ("file1.txt", b"content1"),
        ("file2.txt", b"content2"),
        ("file3.txt", b"content3"),
    ]

    for name, content in files_to_add:
        appender.append_file(name, content)

    sample_zip.seek(0)
    with ZipFile(sample_zip) as zf:
        assert len(zf.namelist()) == 4  # initial + 3 new files
        for name, content in files_to_add:
            assert zf.read(name) == content
        assert zf.read("initial.txt") == b"initial content"


def test_append_multiple_file() -> None:
    with temporary_zip_file() as f:
        test_append_multiple(f)


def test_append_large(sample_zip: BinaryIO) -> None:
    """Test appending a larger file to ensure compression works."""
    large_content: bytes = b"Large content\n" * 1000

    appender: ZipAppender = ZipAppender(sample_zip)
    appender.append_file("large.txt", large_content)

    sample_zip.seek(0)
    with ZipFile(sample_zip) as zf:
        assert zf.read("large.txt") == large_content


def test_append_large_file() -> None:
    with temporary_zip_file() as f:
        test_append_large(f)


def test_zip_integrity(sample_zip: BinaryIO) -> None:
    """Test that the resulting ZIP file maintains integrity after modifications."""
    appender: ZipAppender = ZipAppender(sample_zip)
    appender.append_file("new.txt", b"new content")

    sample_zip.seek(0)
    try:
        with ZipFile(sample_zip) as zf:
            test_result: str | None = zf.testzip()
            assert test_result is None
    except zipfile.BadZipFile:
        pytest.fail("ZIP file integrity check failed")


def test_zip_integrity_file() -> None:
    with temporary_zip_file() as f:
        test_zip_integrity(f)


def test_unicode_filenames(sample_zip: BinaryIO) -> None:
    """Test handling of Unicode filenames."""
    appender: ZipAppender = ZipAppender(sample_zip)
    unicode_filename: str = "файл.txt"
    appender.append_file(unicode_filename, b"unicode content")

    sample_zip.seek(0)
    with ZipFile(sample_zip) as zf:
        namelist: List[str] = zf.namelist()
        assert unicode_filename in namelist
        assert zf.read(unicode_filename) == b"unicode content"


def test_unicode_filenames_file() -> None:
    with temporary_zip_file() as f:
        test_unicode_filenames(f)


def test_empty(sample_zip: BinaryIO) -> None:
    """Test appending an empty file."""
    appender: ZipAppender = ZipAppender(sample_zip)
    appender.append_file("empty.txt", b"")

    sample_zip.seek(0)
    with ZipFile(sample_zip) as zf:
        assert "empty.txt" in zf.namelist()
        assert zf.read("empty.txt") == b""


def test_empty_file() -> None:
    with temporary_zip_file() as f:
        test_empty(f)
