#!/usr/bin/env python3
"""Auto phân tích thị trường crypto (CLI) với chiến lược BUY/SELL + SL/TP.

Nâng cấp:
- Quét rất nhiều coin bằng phân trang CoinGecko (tối đa 1000).
- Confidence model dạng "AI scoring" (xấp xỉ logistic) với nhiều đặc trưng thị trường.
- Trả về confidence_reasons đa dạng theo đóng góp thực tế của từng yếu tố.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

COINGECKO = "https://api.coingecko.com/api/v3"
MAX_PER_PAGE = 250
MAX_TOTAL_TOP = 1000


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
class MarketContext:
    momentum_mean: float
    momentum_std: float
    trend_mean: float
    trend_std: float
    liq_mean: float
    liq_std: float
    vol_mean: float
    vol_std: float
    rank_mean: float
    rank_std: float


@dataclass
class TradePlan:
    symbol: str
    name: str
    side: str
    confidence: float
    confidence_reasons: list[str]
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


def sigmoid(x: float) -> float:
    x = clamp(x, -20, 20)
    return 1 / (1 + math.exp(-x))


def safe_zscore(value: float, mean: float, std: float) -> float:
    if std < 1e-9:
        return 0.0
    return (value - mean) / std


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"accept": "application/json", "user-agent": "crypto-market-agent/4.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def demo_market_data() -> list[CoinSnapshot]:
    base = [
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
    synthetic: list[CoinSnapshot] = []
    for i in range(11, 221):
        ch24 = ((i % 13) - 6) * 0.7 + ((i % 5) - 2) * 0.2
        ch7 = ((i % 17) - 8) * 1.1 + ((i % 7) - 3) * 0.5
        price = max(0.02, 20 - i * 0.05)
        mcap = max(5e7, 4e9 - i * 1.6e7)
        vol = max(8e6, 8e8 - i * 2.3e6)
        synthetic.append(CoinSnapshot(f"ALT{i}", f"Altcoin {i}", price, mcap, i, ch24, ch7, vol))
    return base + synthetic


def _map_market_items(data: list[dict[str, Any]]) -> list[CoinSnapshot]:
    out: list[CoinSnapshot] = []
    for item in data:
        out.append(
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
    return out


def fetch_market_data(vs_currency: str, per_page: int, use_demo: bool = False) -> list[CoinSnapshot]:
    limit = max(1, min(per_page, MAX_TOTAL_TOP))
    if use_demo:
        return demo_market_data()[:limit]

    all_items: list[CoinSnapshot] = []
    page = 1
    while len(all_items) < limit:
        batch_size = min(MAX_PER_PAGE, limit - len(all_items))
        data = get_json(
            f"{COINGECKO}/coins/markets",
            {
                "vs_currency": vs_currency,
                "order": "market_cap_desc",
                "per_page": batch_size,
                "page": page,
                "sparkline": "false",
                "price_change_percentage": "24h,7d",
            },
        )
        if not data:
            break
        all_items.extend(_map_market_items(data))
        page += 1

    return all_items[:limit]


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


def build_market_context(coins: list[CoinSnapshot]) -> MarketContext:
    if not coins:
        return MarketContext(0, 1, 0, 1, 0, 1, 0, 1, 0, 1)

    momentum = [c.price_change_24h_pct for c in coins]
    trend = [c.price_change_7d_pct for c in coins]
    liq = [c.volume_24h / max(c.market_cap, 1) for c in coins]
    vol = [volatility_proxy(c) for c in coins]
    max_rank = max((c.market_cap_rank or len(coins)) for c in coins)
    rank_quality = [1 - ((c.market_cap_rank or max_rank) - 1) / max(1, max_rank - 1) for c in coins]

    def s(values: list[float]) -> tuple[float, float]:
        return statistics.fmean(values), max(statistics.pstdev(values), 1e-9)

    mom_m, mom_s = s(momentum)
    tr_m, tr_s = s(trend)
    liq_m, liq_s = s(liq)
    vol_m, vol_s = s(vol)
    rank_m, rank_s = s(rank_quality)
    return MarketContext(mom_m, mom_s, tr_m, tr_s, liq_m, liq_s, vol_m, vol_s, rank_m, rank_s)


def confidence_engine(coin: CoinSnapshot, pulse: MarketPulse, side: str, ctx: MarketContext) -> tuple[float, list[str]]:
    rel_liq = coin.volume_24h / max(coin.market_cap, 1)
    vol = volatility_proxy(coin)
    rank_quality = 1 - ((coin.market_cap_rank or 1000) - 1) / 999

    sign = 1 if side == "BUY" else -1
    factors = {
        "trend": sign * safe_zscore(coin.price_change_7d_pct, ctx.trend_mean, ctx.trend_std),
        "momentum": sign * safe_zscore(coin.price_change_24h_pct, ctx.momentum_mean, ctx.momentum_std),
        "liquidity": safe_zscore(rel_liq, ctx.liq_mean, ctx.liq_std),
        "stability": -safe_zscore(vol, ctx.vol_mean, ctx.vol_std),
        "rank_quality": safe_zscore(rank_quality, ctx.rank_mean, ctx.rank_std),
        "market_regime": ((pulse.total - 50) / 10.0) * sign,
    }

    weights = {
        "trend": 0.90,
        "momentum": 0.75,
        "liquidity": 0.60,
        "stability": 0.45,
        "rank_quality": 0.40,
        "market_regime": 0.50,
    }

    linear = -0.15
    contributions: dict[str, float] = {}
    for k, v in factors.items():
        c = v * weights[k]
        contributions[k] = c
        linear += c

    probability = sigmoid(linear)
    confidence = clamp(10 + 85 * probability, 10, 95)

    templates = {
        "trend": ("Xu hướng 7d đồng pha với lệnh", "Xu hướng 7d đi ngược lệnh"),
        "momentum": ("Momentum 24h đang ủng hộ entry", "Momentum 24h yếu/đi ngược"),
        "liquidity": ("Thanh khoản tương đối mạnh", "Thanh khoản thấp, dễ trượt giá"),
        "stability": ("Biến động trong vùng dễ quản trị", "Biến động lớn, rủi ro quét SL cao"),
        "rank_quality": ("Vị thế vốn hóa tốt trong market", "Vốn hóa thấp hơn mặt bằng ưu tiên"),
        "market_regime": ("Market regime phù hợp hướng lệnh", "Market regime chưa ủng hộ hướng lệnh"),
    }

    top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    reasons: list[str] = []
    for key, contribution in top:
        pos_msg, neg_msg = templates[key]
        msg = pos_msg if contribution >= 0 else neg_msg
        reasons.append(f"{msg} ({contribution:+.2f})")

    return confidence, reasons


def build_trade_plan(coin: CoinSnapshot, pulse: MarketPulse, ctx: MarketContext) -> TradePlan:
    side = signal_side(coin, pulse)
    vol = volatility_proxy(coin)
    vol_factor = clamp(vol / 100, 0.015, 0.16)

    if side == "BUY":
        entry = coin.current_price * (1 - vol_factor * 0.25)
        stop_loss = entry * (1 - vol_factor * 1.2)
        take_profit_1 = entry * (1 + vol_factor * 1.8)
        take_profit_2 = entry * (1 + vol_factor * 3.2)
        invalidation = "Giá đóng nến H4 dưới SL hoặc volume giảm mạnh trong nhịp hồi."
        note = "Ưu tiên vào lệnh theo từng phần 40%/30%/30%, dời SL về hòa vốn sau khi chạm TP1."
    else:
        entry = coin.current_price * (1 + vol_factor * 0.25)
        stop_loss = entry * (1 + vol_factor * 1.2)
        take_profit_1 = entry * (1 - vol_factor * 1.8)
        take_profit_2 = entry * (1 - vol_factor * 3.2)
        invalidation = "Giá đóng nến H4 trên SL hoặc market breadth cải thiện đột biến."
        note = "Giảm khối lượng short khi funding dương cao; ưu tiên chốt từng phần tại TP1/TP2."

    risk = abs(entry - stop_loss)
    rr1 = abs(take_profit_1 - entry) / risk if risk else 0.0
    rr2 = abs(take_profit_2 - entry) / risk if risk else 0.0
    confidence, reasons = confidence_engine(coin, pulse, side, ctx)

    return TradePlan(
        symbol=coin.symbol,
        name=coin.name,
        side=side,
        confidence=confidence,
        confidence_reasons=reasons,
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
    candidate_limit = max(limit * 6, 180)
    ranked = sorted(coins, key=lambda c: (c.market_cap_rank or 9999, -c.volume_24h))
    selected = ranked[: min(len(ranked), candidate_limit)]
    ctx = build_market_context(coins)
    plans = [build_trade_plan(c, pulse, ctx) for c in selected]
    return sorted(plans, key=lambda p: p.confidence, reverse=True)[: max(1, limit)]


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
    lines.append(f"Phạm vi phân tích: **{len(coins)} coin**")
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

    lines.append("## Chiến lược BUY/SELL + SL/TP (xếp hạng theo confidence AI)")
    for idx, plan in enumerate(plans, start=1):
        lines.append(f"### {idx}) {plan.name} ({plan.symbol}/{quote_asset.upper()}) — {plan.side} | Confidence: {plan.confidence:.1f}/100")
        lines.append(f"- Entry tham chiếu: **{fmt_price(plan.entry)} {currency.upper()}**")
        lines.append(f"- Stop-loss (SL): **{fmt_price(plan.stop_loss)} {currency.upper()}**")
        lines.append(
            f"- Take-profit (TP): **TP1 {fmt_price(plan.take_profit_1)}** | **TP2 {fmt_price(plan.take_profit_2)}** {currency.upper()}"
        )
        lines.append(f"- Risk/Reward: TP1 = {plan.risk_reward_tp1:.2f}R | TP2 = {plan.risk_reward_tp2:.2f}R")
        lines.append(f"- Lý do confidence: {'; '.join(plan.confidence_reasons[:3])}")
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
    parser.add_argument("--top", type=int, default=200, help="Số lượng coin top market cap để phân tích (tối đa 1000)")
    parser.add_argument("--strategy-limit", type=int, default=30, help="Số coin xuất chiến lược giao dịch")
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
            "analyzed_coins": len(coins),
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
