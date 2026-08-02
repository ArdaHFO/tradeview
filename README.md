# US Trading Scanner — intraday + swing

İki ayrı sistem, tek repo:

| | **Swing** (günlük bar) | **Intraday** (dakikalık bar) |
|---|---|---|
| Veri | yfinance — **ücretsiz, anahtarsız**, 10 yıl, 500 sembol | Polygon free tier — 5 istek/dk, tick yok |
| Örneklem | 5.973 işlem | 152 işlem |
| Durum | ✅ **`meanrev` doğrulandı** (örneklem dışı dahil) | ⛔ hiçbir setup doğrulanamadı |

**Hızlı başlangıç (swing — çalışan taraf):**

```bash
pip install -r requirements.txt
python main.py swing-scan --strategy meanrev   # bugün ne alınır
python main.py swing --strategy meanrev --years 10 --oos-from 2023-01-01
```

API anahtarı gerekmez.

---

## Swing sistemi (`src/scanner/swing/`)

Üç **yayınlanmış** strateji, parametreleri bu veriye uydurulmadan olduğu gibi
uygulandı — intraday setupları backtest iyi görünene kadar elle ayarlanmış ve
sonra doğrulamayı geçememişti; bu gürültüye uydurmanın klasik imzası.

10 yıl, 503 sembol (S&P 500), 2016-07 → 2026-07:

| Strateji | İşlem | PF [%95 GA] | Getiri | vs SPY (+305%) | Örneklem dışı | Karar |
|---|---|---|---|---|---|---|
| **`meanrev`** (Connors RSI-2) | 5.973 | **1.28** [1.19–1.38] | **+499%** | ✅ 3/3 eksende | ✅ PF 1.30 | ✅ **doğrulandı** |
| `breakout` (Donchian-55) | 870 | 1.27 [1.05–1.51] | +130% | ❌ SPY'ye yeniliyor | ⚠ tutarsız | ⛔ |
| `trend` (20MA pullback) | 3.103 | 0.83 | −96% | ❌ | — | ⛔ kanıtlanmış zararda |

**`meanrev` örneklem dışı kontrolü** (kesim 2023-01-01) — asıl sınav:

```
IN-SAMPLE :  3519 işlem | beklenti +0.038 R [+0.023, +0.053] | PF 1.26 ✅
OUT-SAMPLE:  2454 işlem | beklenti +0.042 R [+0.025, +0.059] | PF 1.30 ✅
```

Edge, stratejinin seçilmesinde hiç rol oynamamış veride de duruyor.

**SPY al-tut karşılaştırması** — pozitif beklenti "yapmaya değer" demek değildir:

| | meanrev | SPY |
|---|---|---|
| Toplam getiri | +499.2% | +304.6% |
| Max drawdown | **−23.5%** | −33.7% |
| CAGR | 19.6% | 15.0% |
| Calmar | **0.83** | 0.44 |

`breakout` istatistiksel olarak anlamlı bir edge'e sahip **ama** SPY'nin yarısı
kadar getiri veriyor — bu yüzden `benchmark.py` var.

### ⚠ Bilinen kısıt: survivorship bias

Evren **bugünün** S&P 500 listesi. Geçmişe uygulanınca "hangi şirketlerin ayakta
kalacağını önceden bilmek" varsayımı giriyor. Bu **özellikle `meanrev`'i şişirir**:
mean reversion düşeni alır, endeksten atılanlar da tam olarak düşüp toparlanamayanlardır.
Gerçek sonuç raporlanandan **kötüdür**; ne kadar kötü olduğu point-in-time endeks
verisi olmadan ölçülemiyor (ücretsiz değil). Her raporda uyarı olarak basılıyor.

---

## Intraday sistemi

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

**2026-07/08 bulgusu — üç setup da doğrulamayı geçemedi.** Bootstrap + Monte Carlo
(`python main.py validate`, aşağıya bakın) ile ölçüldüğünde:

| Setup | Örneklem | Beklenti (R) | %95 güven aralığı | PF | Karar |
|---|---|---|---|---|---|
| `GAP_AND_GO` | 41 işlem / 13 seans | −0.425 | −0.735 … −0.085 | 0.43 | **kanıtlanmış zararda** |
| `GAP_AND_GO` +gap tavanı | 44 işlem / 22 seans | −0.084 | −0.431 … +0.271 | 0.86 | kanıtlanmadı |
| `ORB` | 65 işlem / 13 seans | −0.376 | −0.660 … −0.055 | 0.54 | **kanıtlanmış zararda** |
| `VWAP_REVERSION` | 152 işlem / 27 seans | +0.061 | −0.134 … **+0.259** | 1.10 | **kanıtlanmadı** |

`MAX_GAP_PCT = 20` tavanı (aşırı esnemiş gapleri elemek) GAP_AND_GO'yu
"kanıtlanmış zararda"dan "kanıtlanmadı"ya taşıdı — PF 0.43 → 0.86, güven aralığı
artık sıfırı içeriyor. Yani filtre **işe yaradı**, ama beklenti hâlâ negatif ve
44 işlem karar vermek için az. Setup kapalı kalıyor.

GAP_AND_GO ve ORB'un aralığı tamamen sıfırın altında: zararları şans değil sistematik.
Her biri için iki parametre düzeltmesi + bir araştırma turu denendi (daha geniş stop,
sert order-flow kapısı, gap tavanı, VWAP yön filtresi); hiçbiri PF'yi 1'in üstüne
çıkarmadı, ORB'un son hâli 30 günde 1 işlem üretti. İkisi de
`cfg.signal.disabled_setups` ile kapatıldı — kod silinmedi.

VWAP_REVERSION zarar etmiyor ama **edge'i de kanıtlanmış değil**: PF 1.10'un güven
aralığı 0.80–1.53, yani başabaşın altı hâlâ makul bir sonuç. Bu etki büyüklüğünü
kanıtlamak için ~1.580 işlem gerekir (elde 152). Simüle koşuların %27'si zararla
bitiyor. **Bu yüzden hiçbir setup canlı paraya hazır değil.**

Muhtemel kök neden: free-tier veride tick yok, bu yüzden order flow (CVD, taker
imbalance) bar şeklinden türetiliyor (`buy_ratio = (close-low)/(high-low)`) — bu da
CVD'yi fiyatla yapısal olarak korelasyonlu yapıp "order flow teyidi"ni bilgi taşımaz
hâle getiriyor. Gerçek tick verisi olmadan bu üç setup'ın adil bir testi yapılamıyor.

Detaylar için `git log --oneline` üzerinde `gap_and_go:`, `orb:`, `validation:`
commit mesajlarına bakın.

**Risk** (`risk.py`): işlem başına %0.5 equity, ATR-stop sizing, günlük -%2
kill-switch, PDT sayacı, kapanışa son 15 dk giriş yasağı.

## Kullanım

```bash
pip install -r requirements.txt
cp .env.example .env        # POLYGON_API_KEY + Telegram bilgilerini doldur

python main.py demo         # API anahtarsız uçtan uca sentetik test
python main.py screen       # Polygon ile gerçek stage-1 watchlist
python main.py live         # watchlist + canlı websocket sinyal akışı (RT plan gerekir)
python main.py dashboard    # tarayıcı arayüzü → http://localhost:8000
```

### Backtest ve istatistiksel doğrulama

```bash
# 30 seans geriye backtest, işlemleri JSON'a yaz
python main.py backtest --days 30 --top 10 --save-trades bt_trades.json

# edge gerçek mi, gürültü mü? (bootstrap + monte carlo)
python main.py validate --trades bt_trades.json
python main.py validate --log backtest_baseline_30d.log --setup VWAP_REVERSION
```

`validate` bir backtest'in profit factor'ünü **nokta tahmini olarak değil güven
aralığıyla** raporlar; aralık başabaşı içeriyorsa edge kanıtlanmamıştır. Ayrıca
gözlenen etki büyüklüğünü kanıtlamak için kaç işlem gerektiğini, alternatif
koşularda drawdown dağılımını ve zararla bitme olasılığını verir. Edge
kanıtlanmadığında çıkış kodu sıfırdan farklıdır, böylece bir pipeline'ı kesebilir.
Aynı özet dashboard'da **Backtest Doğrulama** panelinde equity curve ile birlikte
görünür.

Testler: `python -m pytest tests/ -q` (67 test).

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
├── backtest.py           # bias-free minute-bar replay + fill simülasyonu
├── validation.py         # bootstrap + monte carlo: edge gerçek mi?
├── dashboard.py          # FastAPI arayüz (watchlist, sinyaller, doğrulama)
└── alerts/telegram.py    # Telegram bildirimi
```

## ⚠️ Beklenti yönetimi

Bu bot **sinyal üretir, emir göndermez** ve hiçbir sinyal "kesin" değildir.
Edge istatistikseldir: gerçek parayla kullanmadan önce **backtest + 20-30 seans
paper trading** ile profit factor > 1.3 kanıtlanmalıdır. Her sinyal SQLite'a
kaydedilir (`signals.db`) — performans analizi bu kayıtlar üzerinden yapılır.

**Intraday tarafında hiçbir setup bu barı geçmiş değil** — iki setup kanıtlanmış
şekilde zararda, üçüncüsünün edge'i kanıtlanamadı. Bilinen en büyük kısıt, backtest
verisinde gerçek tick bulunmaması ve order flow'un bar şeklinden türetilmesi.

**Swing tarafında `meanrev` doğrulamayı ve örneklem dışı kontrolü geçti.** Bu
şunu ifade eder: ölçülen edge şansla açıklanamaz. Şunu **ifade etmez**:

- ❌ "Kazanacağı garanti" — %66 win rate demek her 3 işlemden biri zarar demek.
  Backtestte 434 stop-out ve %23.5 drawdown var.
- ❌ "Canlıda aynısı olur" — sonuçlar survivorship bias'lı ve backtest, komisyon/
  slippage modellenmiş olsa bile gerçek fill değildir.
- ❌ "%100 başarılı" — böyle bir şey yok. Edge istatistikseldir: yeterince çok
  işlemde pozitif beklenti, tek tek işlemlerde belirsizlik.

Sıradaki adım backtest değil, **ileriye dönük test**: 20-30 seans paper trading.
Geçmişe bakan hiçbir sayı bunun yerini tutmaz.
