"""Kaggle CLI entry point that prefers IPv4 on networks with broken IPv6 routing."""

from __future__ import annotations

import socket


_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(*args, **kwargs):
    results = _getaddrinfo(*args, **kwargs)
    ipv4 = [result for result in results if result[0] == socket.AF_INET]
    return ipv4 or results


def main() -> None:
    socket.getaddrinfo = _ipv4_getaddrinfo
    from kaggle.cli import main as kaggle_main

    raise SystemExit(kaggle_main())


if __name__ == "__main__":
    main()
