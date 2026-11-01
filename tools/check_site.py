"""Check that the buildless project page has no broken local references."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json
import math


ROOT = Path(__file__).resolve().parents[1] / "site"


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        for key in ("src", "href", "data-full"):
            if attrs.get(key):
                self.refs.append((tag, key, attrs[key]))


def main():
    parser = References()
    parser.feed((ROOT / "index.html").read_text(encoding="utf-8"))
    for name in ("gallery", "csa", "results", "sources"):
        json.loads((ROOT / "data" / f"{name}.json").read_text(encoding="utf-8"))
    gallery = json.loads((ROOT / "data/gallery.json").read_text(encoding="utf-8"))
    assert len(gallery["scenes"]) == 16, "Expected 16 paired scenes"
    for scene in gallery["scenes"]:
        for model in ("drivor", "cofidrive"):
            parser.refs.append(("gallery", model, scene[model]))
    for camera in ("Front", "Left", "Right", "Rear"):
        for model in ("drivor", "cofidrive"):
            panels = gallery["attention"][camera][model]
            assert len(panels) == 6
            parser.refs.extend(("attention", camera, src) for src in panels)
    csa = json.loads((ROOT / "data/csa.json").read_text(encoding="utf-8"))
    assert len(csa) == 5
    for case in csa:
        for key, count in (("original", 64), ("augmented", 192)):
            assert len(case[key]) == count
            for trajectory in case[key]:
                assert len(trajectory) == 8
                assert all(len(p) == 2 and all(math.isfinite(v) for v in p) for p in trajectory)
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    assert ".pdf" not in html.lower(), "Visualizations should open browser images, not PDF files"
    missing = []
    for tag, key, value in parser.refs:
        parts = urlsplit(value)
        if parts.scheme or parts.netloc:
            continue
        if parts.path:
            target = (ROOT / unquote(parts.path)).resolve()
            if not target.is_relative_to(ROOT.resolve()) or not target.is_file():
                missing.append(f"{tag}[{key}]={value}")
        elif parts.fragment and parts.fragment not in parser.ids:
            missing.append(f"{tag}[{key}]={value}")
    if missing:
        raise SystemExit("Missing local references:\n" + "\n".join(missing))
    print(f"OK: {len(parser.refs)} links and image references checked")


if __name__ == "__main__":
    main()
