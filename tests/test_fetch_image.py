import importlib.util
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "fetch_image", Path(__file__).resolve().parent.parent / "scripts" / "fetch_image.py")
fetch_image = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_image)


@pytest.mark.parametrize("html, expected", [
    # ImgBB-style page
    ('<meta property="og:image" content="https://i.ibb.co/abc/photo.jpg" />', "https://i.ibb.co/abc/photo.jpg"),
    # attribute order swapped, twitter tag
    ('<meta content="https://i.imgur.com/x.png" name="twitter:image">', "https://i.imgur.com/x.png"),
    # relative URL and HTML entities
    ('<meta property="og:image" content="/img/a.png?w=1&amp;h=2">', "https://site.example/img/a.png?w=1&h=2"),
    ('<link rel="image_src" href="https://cdn.example/p.webp">', "https://cdn.example/p.webp"),
    ('<html><title>no image here</title></html>', None),
])
def test_find_image_url(html, expected):
    assert fetch_image.find_image_url(html, "https://site.example/page/1") == expected


def test_page_link_is_followed(tmp_path, monkeypatch):
    img = Image.new("RGBA", (40, 30), (255, 0, 0, 0))
    buf = tmp_path / "x.png"
    img.save(buf)
    pages = {
        "https://ibb.co/r25mnsGd": (b'<meta property="og:image" content="https://i.ibb.co/r/x.png">', "text/html"),
        "https://i.ibb.co/r/x.png": (buf.read_bytes(), "image/png"),
    }
    monkeypatch.setattr(fetch_image, "fetch", lambda url: (*pages[url], url))
    out = tmp_path / "input.jpg"
    fetch_image.main("https://ibb.co/r25mnsGd", out)
    saved = Image.open(out)
    assert saved.size == (40, 30) and saved.mode == "RGB"
    assert saved.getpixel((0, 0)) == (255, 255, 255)   # transparent -> white, like a photo


def test_page_without_image_gives_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_image, "fetch", lambda url: (b"<html></html>", "text/html", url))
    with pytest.raises(SystemExit, match="Copy image address"):
        fetch_image.main("https://example.com/album", tmp_path / "x.jpg")
