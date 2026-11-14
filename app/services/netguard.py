"""Keeps server-side browsing on the public internet.

On a hosted deployment, any URL a user submits is opened by a browser inside the server's network, so private,
loopback and link-local destinations (internal services, cloud metadata at 169.254.169.254, ...) are refused, both
for the URLs we check up front and for every request a browser context makes, including redirects. Set
ALLOW_PRIVATE_URLS=true for local development (for example to record an app on localhost).

Residual risk: the browser resolves DNS itself, so a host that changes its answer between our check and the
browser's lookup (DNS rebinding) is not fully covered.
"""
import asyncio
import ipaddress
import socket
import time
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

from playwright.async_api import BrowserContext, Route

from app.config import env_flag

CACHE_TTL_SEC = 60
_cache: Dict[str, Tuple[Optional[str], float]] = {}


class BlockedURL(ValueError):
    pass


def _public_ip(text: str) -> bool:
    addr = ipaddress.ip_address(text.split("%", 1)[0])
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_global and not addr.is_multicast


async def host_problem(host: str) -> Optional[str]:
    """None when every address of `host` is public, otherwise a plain-language reason."""
    if env_flag("ALLOW_PRIVATE_URLS"):
        return None
    host = (host or "").strip("[]").lower().rstrip(".")
    if not host:
        return "the URL has no host"
    cached = _cache.get(host)
    if cached and cached[1] > time.monotonic():
        return cached[0]
    try:
        ips = [host] if _is_ip(host) else sorted({info[4][0] for info in await asyncio.get_running_loop().getaddrinfo(
            host, None, type=socket.SOCK_STREAM)})
        problem = None if ips and all(_public_ip(ip) for ip in ips) else f"{host} points to a private network address"
    except (OSError, ValueError):
        problem = f"{host} could not be resolved"
    _cache[host] = (problem, time.monotonic() + CACHE_TTL_SEC)
    return problem


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        return True
    except ValueError:
        return False


async def ensure_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise BlockedURL("Only http and https URLs can be opened")
    problem = await host_problem(parsed.hostname or "")
    if problem:
        raise BlockedURL(f"{url} can't be opened: {problem}")


async def guard_context(context: BrowserContext) -> None:
    """Aborts every request (navigations, redirects, subresources) that leaves the public internet."""
    if env_flag("ALLOW_PRIVATE_URLS"):
        return

    async def check(route: Route) -> None:
        parsed = urlparse(route.request.url)
        if parsed.scheme in ("http", "https", "ws", "wss") and await host_problem(parsed.hostname or ""):
            await route.abort("blockedbyclient")
        else:
            await route.continue_()

    await context.route("**/*", check)
