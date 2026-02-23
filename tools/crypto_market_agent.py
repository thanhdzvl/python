#!/usr/bin/env python3
"""Auto phân tích thị trường crypto (CLI) với chiến lược BUY/SELL + SL/TP.

- Nguồn dữ liệu chính: CoinGecko public API.
- Có chế độ --demo để chạy offline khi môi trường bị chặn outbound network.
- Kết quả gồm market pulse và chiến lược giao dịch mẫu (không phải lời khuyên đầu tư).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

COINGECKO = "https://api.coingecko.com/api/v3"


@dataclass
class CoinSnapshot:
    symbol: str
    name: str
    current_price: float
    market_cap: float
    market_cap_rank: int | None
    price_change_24h_pct: float
    price_change_7d_pct: float
    volume_24h: float


@dataclass
class MarketPulse:
    trend_score: float
    volatility_score: float
    liquidity_score: float
    breadth_score: float

    @property
    def total(self) -> float:
        return (self.trend_score + self.volatility_score + self.liquidity_score + self.breadth_score) / 4


@dataclass
class TradePlan:
    symbol: str
    name: str
    side: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_tp1: float
    risk_reward_tp2: float
    invalidation: str
    note: str


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"accept": "application/json", "user-agent": "crypto-market-agent/2.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def demo_market_data() -> list[CoinSnapshot]:
    return [
        CoinSnapshot("BTC", "Bitcoin", 95000, 1.85e12, 1, 2.3, 8.5, 4.1e10),
        CoinSnapshot("ETH", "Ethereum", 5100, 6.4e11, 2, 1.8, 7.2, 2.5e10),
        CoinSnapshot("BNB", "BNB", 820, 1.2e11, 5, 0.9, 4.0, 1.8e9),
        CoinSnapshot("SOL", "Solana", 230, 1.05e11, 6, 4.9, 11.4, 5.4e9),
        CoinSnapshot("XRP", "XRP", 1.18, 6.7e10, 7, -0.5, 3.1, 2.4e9),
        CoinSnapshot("DOGE", "Dogecoin", 0.26, 3.8e10, 9, -1.2, 2.0, 1.7e9),
        CoinSnapshot("ADA", "Cardano", 0.79, 2.8e10, 10, -0.2, 1.1, 1.2e9),
        CoinSnapshot("AVAX", "Avalanche", 48, 1.9e10, 12, 3.6, 9.4, 9.8e8),
        CoinSnapshot("DOT", "Polkadot", 9.4, 1.5e10, 15, -2.8, -4.2, 7.1e8),
        CoinSnapshot("LINK", "Chainlink", 27.5, 1.7e10, 14, 2.0, 6.6, 8.2e8),
    ]


def fetch_market_data(vs_currency: str, per_page: int, use_demo: bool = False) -> list[CoinSnapshot]:
    if use_demo:
        return demo_market_data()[:per_page]

    data = get_json(
        f"{COINGECKO}/coins/markets",
        {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h,7d",
        },
    )
    snapshots: list[CoinSnapshot] = []
    for item in data:
        snapshots.append(
            CoinSnapshot(
                symbol=item.get("symbol", "").upper(),
                name=item.get("name", "Unknown"),
                current_price=float(item.get("current_price") or 0.0),
                market_cap=float(item.get("market_cap") or 0.0),
                market_cap_rank=item.get("market_cap_rank"),
                price_change_24h_pct=float(item.get("price_change_percentage_24h") or 0.0),
                price_change_7d_pct=float(item.get("price_change_percentage_7d_in_currency") or 0.0),
                volume_24h=float(item.get("total_volume") or 0.0),
            )
        )
    return snapshots


def compute_pulse(coins: list[CoinSnapshot]) -> MarketPulse:
    if not coins:
        return MarketPulse(0, 0, 0, 0)

    avg_24h = statistics.fmean(c.price_change_24h_pct for c in coins)
    avg_7d = statistics.fmean(c.price_change_7d_pct for c in coins)
    stdev_24h = statistics.pstdev(c.price_change_24h_pct for c in coins)
    advancers = sum(1 for c in coins if c.price_change_24h_pct > 0)

    total_mcap = sum(c.market_cap for c in coins) or 1.0
    total_vol = sum(c.volume_24h for c in coins)
    volume_ratio = total_vol / total_mcap

    trend_score = clamp(50 + avg_24h * 2 + avg_7d * 0.8, 0, 100)
    volatility_score = clamp(100 - stdev_24h * 4, 0, 100)
    # Thanh khoản thường dao động thấp theo tỷ lệ volume/marketcap, scale tuyến tính để dễ đọc hơn.
    liquidity_score = clamp(volume_ratio * 600, 0, 100)
    breadth_score = clamp((advancers / len(coins)) * 100, 0, 100)
    return MarketPulse(trend_score, volatility_score, liquidity_score, breadth_score)


def classify_market(score: float) -> str:
    if score >= 70:
        return "🟢 Bullish"
    if score >= 55:
        return "🟡 Nghiêng tăng"
    if score >= 45:
        return "🟠 Sideway"
    return "🔴 Bearish"


def top_movers(coins: list[CoinSnapshot], n: int = 5) -> tuple[list[CoinSnapshot], list[CoinSnapshot]]:
    gainers = sorted(coins, key=lambda c: c.price_change_24h_pct, reverse=True)[:n]
    losers = sorted(coins, key=lambda c: c.price_change_24h_pct)[:n]
    return gainers, losers


def volatility_proxy(coin: CoinSnapshot) -> float:
    return max(abs(coin.price_change_24h_pct), abs(coin.price_change_7d_pct) / 3)


def signal_side(coin: CoinSnapshot, pulse: MarketPulse) -> str:
    bullish_coin = coin.price_change_24h_pct > 0 and coin.price_change_7d_pct > 0
    bearish_coin = coin.price_change_24h_pct < 0 and coin.price_change_7d_pct < 0

    if pulse.total >= 55 and bullish_coin:
        return "BUY"
    if pulse.total < 45 and bearish_coin:
        return "SELL"
    if bearish_coin and pulse.total < 55:
        return "SELL"
    return "BUY"


def build_trade_plan(coin: CoinSnapshot, pulse: MarketPulse) -> TradePlan:
    side = signal_side(coin, pulse)
    vol = volatility_proxy(coin)
    vol_factor = clamp(vol / 100, 0.015, 0.14)

    if side == "BUY":
        entry = coin.current_price * (1 - vol_factor * 0.25)
        stop_loss = entry * (1 - vol_factor * 1.2)
        take_profit_1 = entry * (1 + vol_factor * 1.8)
        take_profit_2 = entry * (1 + vol_factor * 3.0)
        invalidation = "Giá đóng nến H4 dưới SL hoặc volume giảm mạnh trong nhịp hồi."
        note = "Ưu tiên vào lệnh theo từng phần 40%/30%/30%, dời SL về hòa vốn sau khi chạm TP1."
    else:
        entry = coin.current_price * (1 + vol_factor * 0.25)
        stop_loss = entry * (1 + vol_factor * 1.2)
        take_profit_1 = entry * (1 - vol_factor * 1.8)
        take_profit_2 = entry * (1 - vol_factor * 3.0)
        invalidation = "Giá đóng nến H4 trên SL hoặc market breadth cải thiện đột biến."
        note = "Giảm khối lượng short khi funding dương cao; ưu tiên chốt từng phần tại TP1/TP2."

    risk = abs(entry - stop_loss)
    rr1 = abs(take_profit_1 - entry) / risk if risk else 0.0
    rr2 = abs(take_profit_2 - entry) / risk if risk else 0.0

    trend_alignment = 1 if (coin.price_change_7d_pct >= 0 and side == "BUY") or (coin.price_change_7d_pct < 0 and side == "SELL") else 0
    confidence = clamp(
        45
        + (pulse.total - 50) * (0.8 if side == "BUY" else -0.5)
        + trend_alignment * 8
        + (coin.price_change_24h_pct * (1.2 if side == "BUY" else -1.2)),
        20,
        90,
    )

    return TradePlan(
        symbol=coin.symbol,
        name=coin.name,
        side=side,
        confidence=confidence,
        entry=entry,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        risk_reward_tp1=rr1,
        risk_reward_tp2=rr2,
        invalidation=invalidation,
        note=note,
    )


def generate_trade_plans(coins: list[CoinSnapshot], pulse: MarketPulse, limit: int) -> list[TradePlan]:
    ranked = sorted(coins, key=lambda c: (c.market_cap_rank or 9999, -c.volume_24h))
    selected = ranked[: max(1, limit)]
    return [build_trade_plan(c, pulse) for c in selected]


def format_coin_line(coin: CoinSnapshot, currency: str, quote_asset: str) -> str:
    return (
        f"- {coin.name} ({coin.symbol}/{quote_asset.upper()}) | "
        f"Giá: {coin.current_price:,.4f} {currency.upper()} | "
        f"24h: {coin.price_change_24h_pct:+.2f}% | "
        f"7d: {coin.price_change_7d_pct:+.2f}%"
    )


def fmt_price(v: float) -> str:
    return f"{v:,.6f}".rstrip("0").rstrip(".")


def generate_report(coins: list[CoinSnapshot], currency: str, strategy_limit: int, quote_asset: str) -> str:
    pulse = compute_pulse(coins)
    gainers, losers = top_movers(coins)
    plans = generate_trade_plans(coins, pulse, strategy_limit)
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append(f"# Crypto Market Brief ({now})")
    lines.append("")
    lines.append(f"Tổng điểm thị trường: **{pulse.total:.1f}/100** - {classify_market(pulse.total)}")
    lines.append("")
    lines.append("## Điểm thành phần")
    lines.append(f"- Trend score: {pulse.trend_score:.1f}")
    lines.append(f"- Volatility score: {pulse.volatility_score:.1f} (càng cao càng ổn định)")
    lines.append(f"- Liquidity score: {pulse.liquidity_score:.1f}")
    lines.append(f"- Breadth score: {pulse.breadth_score:.1f} (% coin tăng trong 24h)")
    lines.append("")

    lines.append("## Top tăng 24h")
    for c in gainers:
        lines.append(format_coin_line(c, currency, quote_asset))
    lines.append("")

    lines.append("## Top giảm 24h")
    for c in losers:
        lines.append(format_coin_line(c, currency, quote_asset))
    lines.append("")

    lines.append("## Chiến lược BUY/SELL + SL/TP (tự động)")
    for idx, plan in enumerate(plans, start=1):
        lines.append(f"### {idx}) {plan.name} ({plan.symbol}/{quote_asset.upper()}) — {plan.side} | Confidence: {plan.confidence:.1f}/100")
        lines.append(f"- Entry tham chiếu: **{fmt_price(plan.entry)} {currency.upper()}**")
        lines.append(f"- Stop-loss (SL): **{fmt_price(plan.stop_loss)} {currency.upper()}**")
        lines.append(
            f"- Take-profit (TP): **TP1 {fmt_price(plan.take_profit_1)}** | **TP2 {fmt_price(plan.take_profit_2)}** {currency.upper()}"
        )
        lines.append(f"- Risk/Reward: TP1 = {plan.risk_reward_tp1:.2f}R | TP2 = {plan.risk_reward_tp2:.2f}R")
        lines.append(f"- Điều kiện vô hiệu setup: {plan.invalidation}")
        lines.append(f"- Kế hoạch quản trị lệnh: {plan.note}")
        lines.append("")

    lines.append("## Quản trị vốn đề xuất")
    lines.append("- Rủi ro mỗi lệnh: 0.5% - 1.5% tổng tài khoản.")
    lines.append("- Không mở quá 3 vị thế tương quan cùng chiều.")
    lines.append("- Nếu chạm 3 SL liên tiếp: dừng giao dịch và đánh giá lại điều kiện thị trường.")
    lines.append("")
    lines.append("> Miễn trừ trách nhiệm: kết quả chỉ nhằm mục đích tham khảo/giáo dục, không phải lời khuyên đầu tư.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto phân tích thị trường crypto + chiến lược BUY/SELL/SL/TP.")
    parser.add_argument("--currency", default="usd", help="Đơn vị giá, ví dụ: usd, vnd, eur")
    parser.add_argument("--top", type=int, default=50, help="Số lượng coin top market cap để phân tích")
    parser.add_argument("--strategy-limit", type=int, default=5, help="Số coin xuất chiến lược giao dịch")
    parser.add_argument("--output", choices=["markdown", "json"], default="markdown", help="Định dạng đầu ra")
    parser.add_argument("--quote", default="usdt", help="Quote asset hiển thị cặp giao dịch, ví dụ: usdt")
    parser.add_argument("--demo", action="store_true", help="Dùng dữ liệu mẫu offline (không gọi API)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        coins = fetch_market_data(vs_currency=args.currency, per_page=args.top, use_demo=args.demo)
    except urllib.error.URLError as exc:
        print(f"Lỗi kết nối API: {exc}", file=sys.stderr)
        print("Gợi ý: thêm --demo để chạy bằng dữ liệu mẫu offline.", file=sys.stderr)
        return 2

    pulse = compute_pulse(coins)
    plans = generate_trade_plans(coins, pulse, args.strategy_limit)

    if args.output == "json":
        payload = {
            "generated_at": dt.datetime.utcnow().isoformat() + "Z",
            "currency": args.currency,
            "market_state": classify_market(pulse.total),
            "quote_asset": args.quote.upper(),
            "pulse": asdict(pulse),
            "trade_plans": [asdict(p) for p in plans],
            "coins": [asdict(c) for c in coins],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(generate_report(coins, args.currency, args.strategy_limit, args.quote))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
