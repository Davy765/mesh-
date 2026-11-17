"""Small helpers shared by the public web pages (status page, faucet, playground)."""
import http.server


def _client_ip(handler, trust_proxy):
    """The visitor's address. Behind a reverse proxy (Caddy, nginx) every request comes from the proxy, so
    with --trust-proxy we take the address the proxy recorded (the last X-Forwarded-For entry)."""
    if trust_proxy:
        fwd = handler.headers.get("X-Forwarded-For", "")
        if fwd:
            return fwd.split(",")[-1].strip()
    return handler.client_address[0]


def _web_server(host, port, handler_cls):
    """A threaded HTTP server with the same limits as the nodes: capped concurrency, per-address rate,
    timeouts for slow clients."""
    from .transport import SOCKET_TIMEOUT_S, _LimitedServerMixin, _Limits

    class Server(_LimitedServerMixin, http.server.ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True
        limits = _Limits()
    handler_cls.timeout = SOCKET_TIMEOUT_S
    return Server((host, port), handler_cls)
