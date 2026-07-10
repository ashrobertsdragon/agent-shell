"""Network context provider — reports listening ports and basic connectivity."""

import re
import socket

from agentsh.context.providers import register
from agentsh.models import ContextFragment
from agentsh.shell.protocol import Shell

_PORT_RE = re.compile(r"[:.](\d+)$")


def _has_route_via(family: socket.AddressFamily, address: str) -> bool:
    """Return True if the kernel has a non-loopback route to address.

    Connecting a ``SOCK_DGRAM`` socket only asks the kernel to resolve
    which local interface/source address its routing table would use
    for the given (literal, so no DNS lookup) destination -- no packet
    is ever put on the wire.
    """
    with socket.socket(family, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect((address, 80))
        except OSError:
            return False
        source_address = probe.getsockname()[0]
        return not (
            source_address.startswith("127.") or source_address == "::1"
        )


def _has_active_network_route() -> bool:
    """Return True if the host has a working non-loopback network route.

    This is a local-only, zero-network-I/O approximation of
    "connectivity", not a reachability probe -- see ``_has_route_via``.
    That makes it effectively instant, unlike a ping or DNS query,
    which is what makes it safe to run inside the tight per-provider
    timeout described in ``NetworkProvider.collect``.

    IPv4 is tried first since it's the common case; an IPv6-only host
    (no IPv4 route at all) falls back to an IPv6 destination so it
    isn't misreported as having no network.
    """
    return _has_route_via(socket.AF_INET, "8.8.8.8") or _has_route_via(
        socket.AF_INET6, "2001:4860:4860::8888"
    )


@register("network")
class NetworkProvider:
    """Collects currently listening TCP ports and basic connectivity."""

    name = "network"

    async def collect(self, shell: Shell) -> ContextFragment | None:
        """Return listening ports and connectivity, or None if unavailable.

        ``netstat -an`` is used rather than ``ss`` because ``ss`` is
        Linux-only, while ``netstat`` (with some dialect differences) is
        present on Linux, macOS, and Windows -- the one listing tool
        actually available on every platform this project supports. No
        stderr redirection is used: ``CommandResult`` already separates
        stdout/stderr/exit_code regardless of what the command does with
        fd 2, and POSIX-only redirection syntax such as ``2>/dev/null``
        breaks on cmd.exe and PowerShell.

        Parsing works around real cross-platform format differences
        rather than branching per OS. GNU (Linux) and BSD (macOS)
        netstat both include ``Recv-Q``/``Send-Q`` columns before
        ``Local Address``, while Windows' netstat has no such columns,
        so ``Local Address`` sits at a different column index on
        Windows than on Linux/macOS -- a fixed-column parse would need
        an OS branch. Instead, each whitespace-split token on a
        candidate line is checked for a trailing ``:port`` or ``.port``
        suffix (BSD netstat separates address and port with a dot,
        e.g. ``127.0.0.1.8080``, where GNU/Windows use a colon); the
        first token that matches is the local address's port,
        regardless of which column it lands in. Candidate lines are
        identified by the substring ``"LISTEN"``, which appears in
        Linux's ``LISTEN``, Windows' ``LISTENING``, and BSD's
        ``(LISTEN)`` alike.

        Lines whose first token is ``unix`` are skipped outright: Linux's
        ``netstat -an`` also lists UNIX domain sockets in a separate
        section whose ``State`` column value ``LISTENING`` contains the
        substring ``LISTEN``, and those rows have no ``:port``/``.port``
        local address at all -- some socket paths can still end in a
        dot-number segment (e.g. an abstract name with a trailing PID),
        which would otherwise be misparsed as a TCP port.

        Only TCP listeners are reported: UDP sockets have no state
        column in any of the three netstat dialects, so reliably
        distinguishing "bound" UDP sockets from other UDP rows without
        per-OS parsing isn't feasible in the same pass. This is a
        deliberate scope limitation, not an oversight.

        Only the port and protocol are surfaced in the payload -- the
        bound host portion of the local address (e.g. a real LAN IP)
        is dropped, so this provider cannot leak internal network
        topology into the LLM's context, matching the minimal-
        disclosure precedent set by the environment provider's
        variable allowlist (see docs/security.md).

        Connectivity is intentionally not verified with an active probe
        (ping or DNS resolution): context providers run under a
        per-provider ``asyncio.wait_for`` enforcing
        ``ContextConfig.timeout_ms`` (default 200ms, see
        ``src/agentsh/config.py`` and ``ContextBuilder.build`` in
        ``src/agentsh/context/builder.py``), and a real round trip could
        easily exceed that budget, forcing a shell reset. Instead,
        connectivity is approximated locally and near-instantly via
        ``_has_active_network_route``, which reports only whether the
        host has a non-loopback route -- not whether any specific
        remote host is reachable.
        """
        result = await shell.execute("netstat -an")
        if result.exit_code != 0:
            return None

        ports: list[dict[str, str | int]] = []
        for line in result.stdout.splitlines():
            if "LISTEN" not in line:
                continue
            tokens = line.split()
            if not tokens or tokens[0].lower() == "unix":
                continue
            port = next(
                (m.group(1) for t in tokens if (m := _PORT_RE.search(t))),
                None,
            )
            if port is None:
                continue
            ports.append({"proto": tokens[0].lower(), "port": int(port)})

        network_up = _has_active_network_route()

        return ContextFragment(
            provider=self.name,
            summary=(
                f"{len(ports)} listening port(s), "
                f"network {'up' if network_up else 'down'}"
            ),
            payload={"listening_ports": ports, "network_up": network_up},
        )
