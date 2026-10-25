"""LCM bus URLs.

Camera images are large (a 640x360 RGB-D frame is ~1.6 MB). On a multicast
bus every process receives every packet, subscribed or not, so a controller
on the same bus as the cameras has to drain ~25 MB/s through a small kernel
UDP buffer (212 kB by default on Linux): its own small messages get dropped.
Putting the images on a second bus avoids that.

``resolve_lcm_url(spec)``:

* ``""`` / None   -> None (lcm.LCM() default: $LCM_DEFAULT_URL or the LCM default)
* ``"default+N"`` -> the default URL with its port + N (so a private test bus
  from $LCM_DEFAULT_URL gets a private camera bus too)
* anything else   -> used as is (e.g. ``udpm://239.255.76.67:7668?ttl=0``)
"""

from __future__ import annotations

import os
import re

LCM_DEFAULT = "udpm://239.255.76.67:7667?ttl=0"


def resolve_lcm_url(spec: str | None) -> str | None:
    if not spec:
        return None
    m = re.fullmatch(r"default\+(\d+)", str(spec).strip())
    if not m:
        return str(spec)
    base = os.environ.get("LCM_DEFAULT_URL", "") or LCM_DEFAULT
    pm = re.match(r"^(udpm://[^:/?]+):(\d+)(.*)$", base)
    if not pm:
        raise ValueError(f"cannot offset the port of LCM URL {base!r}")
    return f"{pm.group(1)}:{int(pm.group(2)) + int(m.group(1))}{pm.group(3)}"


def make_lcm(spec: str | None = None):
    import lcm

    url = resolve_lcm_url(spec)
    return lcm.LCM(url) if url else lcm.LCM()
