"""Check that the buildless project page has no broken local references."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


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
