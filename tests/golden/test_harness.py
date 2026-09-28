import socket

import pytest

from tests.golden.adapters import AdapterUnavailable, load_adapter


def test_outbound_connections_are_refused_before_any_packet_is_sent():
    with pytest.raises(RuntimeError, match="loopback"):
        socket.create_connection(("192.0.2.1", 443), timeout=1)


def test_loopback_connections_are_allowed():
    listener = socket.create_server(("127.0.0.1", 0))
    with listener, socket.create_connection(listener.getsockname(), timeout=1):
        pass


def test_missing_implementations_and_capabilities_are_reported_as_unavailable():
    with pytest.raises(AdapterUnavailable):
        load_adapter("parser", "no_such_implementation")
    with pytest.raises(AdapterUnavailable):
        load_adapter("no_such_capability", "legacy")
