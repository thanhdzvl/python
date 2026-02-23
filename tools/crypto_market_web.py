#!/usr/bin/env python3
"""Web GUI cho crypto market agent.

Chạy:
    python3 tools/crypto_market_web.py --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from crypto_market_agent import (
    classify_market,
    compute_pulse,
    fetch_market_data,
    generate_trade_plans,
)

BASE_DIR = Path(__file__).resolve().parent
UI_PATH = BASE_DIR / "crypto_market_ui.html"


class CryptoHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(UI_PATH.read_text(encoding="utf-8"))
            return

        if parsed.path == "/api/analyze":
            query = parse_qs(parsed.query)
            currency = query.get("currency", ["usd"])[0]
            top = int(query.get("top", ["50"])[0])
            strategy_limit = int(query.get("strategy_limit", ["5"])[0])
            quote_asset = query.get("quote", ["usdt"])[0].upper()
            use_demo = query.get("demo", ["false"])[0].lower() == "true"

            try:
                coins = fetch_market_data(currency, top, use_demo=use_demo)
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    {
                        "ok": False,
                        "error": str(exc),
                        "hint": "Nếu môi trường chặn API, bật Demo mode.",
                    },
                    status=HTTPStatus.BAD_GATEWAY,
                )
                return

            pulse = compute_pulse(coins)
            plans = generate_trade_plans(coins, pulse, strategy_limit)
            data = {
                "ok": True,
                "market_state": classify_market(pulse.total),
                "quote_asset": quote_asset,
                "pulse": {
                    "trend": round(pulse.trend_score, 2),
                    "volatility": round(pulse.volatility_score, 2),
                    "liquidity": round(pulse.liquidity_score, 2),
                    "breadth": round(pulse.breadth_score, 2),
                    "total": round(pulse.total, 2),
                },
                "plans": [
                    {
                        "symbol": p.symbol,
                        "pair": f"{p.symbol}/{quote_asset}",
                        "name": p.name,
                        "side": p.side,
                        "confidence": round(p.confidence, 2),
                        "entry": p.entry,
                        "sl": p.stop_loss,
                        "tp1": p.take_profit_1,
                        "tp2": p.take_profit_2,
                        "rr1": round(p.risk_reward_tp1, 2),
                        "rr2": round(p.risk_reward_tp2, 2),
                        "invalidation": p.invalidation,
                        "note": p.note,
                    }
                    for p in plans
                ],
            }
            self._send_json(data)
            return

        self._send_json({"ok": False, "error": "Not found"}, status=HTTPStatus.NOT_FOUND)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto market web GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not UI_PATH.exists():
        raise FileNotFoundError(f"Missing UI file: {UI_PATH}")

    server = ThreadingHTTPServer((args.host, args.port), CryptoHandler)
    print(f"Serving Crypto GUI at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
