"""Download an image for the end-to-end workflow and save it as a JPEG photo.

Accepts a direct image URL or an image *page* URL (ImgBB, Imgur, Google Photos, ...):
for a page, the real image is taken from its og:image / twitter:image tag.

    python scripts/fetch_image.py URL out.jpg
"""

import io
import re
import sys
import urllib.request
from html import unescape
from urllib.parse import urljoin

from PIL import Image, UnidentifiedImageError

UA = "Mozilla/5.0 (X11; Linux x86_64) create3d"
META_PATTERNS = [
    r'<meta[^>]+(?:property|name)=["\'](?:og:image(?::secure_url)?|og:image:url|twitter:image(?::src)?)["\'][^>]*>',
    r'<link[^>]+rel=["\']image_src["\'][^>]*>',
]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read(), resp.headers.get("Content-Type", ""), resp.geturl()


def find_image_url(html, base_url):
    """Return the image URL advertised by an HTML page, or None."""
    for pattern in META_PATTERNS:
        for tag in re.findall(pattern, html, flags=re.IGNORECASE):
            m = re.search(r'(?:content|href)=["\']([^"\']+)["\']', tag, flags=re.IGNORECASE)
            if m:
                return urljoin(base_url, unescape(m.group(1)))
    return None


def open_image(data):
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except (UnidentifiedImageError, OSError):
        return None


def to_photo(img):
    """Flatten onto white and return RGB, like a normal photo (so background removal runs)."""
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, "white")
    return Image.alpha_composite(bg, img).convert("RGB")


def main(url, out):
    data, ctype, final_url = fetch(url)
    img = open_image(data)
    if img is None:
        image_url = find_image_url(data.decode("utf-8", "replace"), final_url)
        if not image_url:
            sys.exit(f"'{url}' is not an image and the page has no image link (og:image). "
                     "Use the direct image link, e.g. right-click the image -> 'Copy image address'.")
        print(f"page link: using its image {image_url}")
        data, ctype, _ = fetch(image_url)
        img = open_image(data)
        if img is None:
            sys.exit(f"Could not read an image from {image_url} (content type: {ctype or 'unknown'}).")
    print(f"input: {img.size[0]}x{img.size[1]} {img.mode}")
    to_photo(img).save(out, quality=95)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
