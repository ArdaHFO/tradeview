# US Day-Trading Scanner

İki aşamalı huni: **~8000 US hissesi → 10-30 "in play" watchlist → order-flow teyitli sinyaller.**

## Strateji

**Aşama 1 — Screener** (`stage1_screener.py`): RVOL ≥ 2, |gap| ≥ %4, $2–50 fiyat,
float < 50M, günlük range ≥ %4 → heat skoruna göre sıralı watchlist.

**Aşama 2 — Sinyal motoru** (`engine.py`): watchlist'teki her sembolde 4 ortogonal
aile (konum/VWAP, **order flow/CVD**, yapı, volatilite) ile confluence skorlaması
(eşik 0.65) — 3 setup şablonu:

| Setup | Pencere | Tetik | Durum |
|---|---|---|---|
| `GAP_AND_GO` | İlk 60 dk | Gap + VWAP üstü + CVD pozitif/artan + 5dk-high kırılımı | ⛔ varsayılan kapalı |
| `VWAP_REVERSION` | 30 dk sonrası | VWAP'tan ≥2 ATR + CVD divergence + tape yavaşlama | ✅ aktif |
| `ORB` | 15 dk sonrası | Opening range kırılımı + hacim spike + CVD teyidi | ⛔ varsayılan kapalı |

**2026-07 backtest bulgusu:** aynı 15 günlük pencerede (161 işlem taban) GAP_AND_GO ve
ORB tutarlı şekilde zararda çıktı (sırasıyla PF 0.35-0.46, iki ayrı parametre
düzeltmesinden sonra da), VWAP_REVERSION ise tek başına PF 1.23 (55 işlem, %51 win)
verdi. İkisi de `cfg.signal.disabled_setups` üzerinden varsayılan olarak kapatıldı —
kod silinmedi, farklı bir piyasa rejiminde tekrar denenebilir. Detaylar için
`git log --oneline` üzerinde `gap_and_go:` ve `orb:` commit mesajlarına bakın.

**Risk** (`risk.py`): işlem başına %0.5 equity, ATR-stop sizing, günlük -%2
kill-switch, PDT sayacı, kapanışa son 15 dk giriş yasağı.

## Kullanım

```bash
pip install -r requirements.txt
cp .env.example .env        # POLYGON_API_KEY + Telegram bilgilerini doldur

python main.py demo         # API anahtarsız uçtan uca sentetik test
python main.py screen       # Polygon ile gerçek stage-1 watchlist
python main.py live         # watchlist + canlı websocket sinyal akışı (RT plan gerekir)
```

Testler: `python -m pytest tests/ -q` (42 test).

## Mimari

```
src/scanner/
├── config.py             # env-driven ayarlar (filtreler, ağırlıklar, risk)
├── session.py            # NY saat dilimi, RTH, lockout pencereleri
├── models.py             # Bar/TradeTick/Quote/Signal veri modelleri
├── stage1_screener.py    # evren taraması → watchlist
├── data/provider.py      # Polygon REST + websocket adaptörü
├── features/
│   ├── indicators.py     # EMA, RSI(Wilder), ATR(Wilder), OR, RVOL, divergence
│   ├── vwap.py           # session-anchored VWAP + bantlar
│   └── orderflow.py      # CVD, taker imbalance, tape hızı, büyük printler
├── strategies/           # gap_and_go, vwap_reversion, orb
├── signal/               # confluence scorer + SQLite recorder
├── risk.py               # sizing, kill-switch, PDT
├── engine.py             # veri yönlendirme + değerlendirme döngüsü
└── alerts/telegram.py    # Telegram bildirimi
```

## ⚠️ Beklenti yönetimi

Bu bot **sinyal üretir, emir göndermez** ve hiçbir sinyal "kesin" değildir.
Edge istatistikseldir: gerçek parayla kullanmadan önce **backtest + 20-30 seans
paper trading** ile profit factor > 1.3 kanıtlanmalıdır. Her sinyal SQLite'a
kaydedilir (`signals.db`) — performans analizi bu kayıtlar üzerinden yapılır.
