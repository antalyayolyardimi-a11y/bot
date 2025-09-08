# Bot Mimari ve Sinyal Üretim Analizi

Bu doküman README içindeki monolit kodun incelenmesi sonucu çıkarılan mimari bileşenleri ve web uygulamasına aktarım stratejisini özetler.

## 1. Veri Kaynağı
- **Borsa:** KuCoin (`https://api.kucoin.com`)
- **Endpointler:** `get_symbol_list()`, `get_all_tickers()`, `get_kline(symbol, interval, limit)`
- **Zaman Dilimleri:** 15m (LTF) ve 1H (HTF)
- Likidite filtresi: quote = USDT (varsayılan), `volValue >= MIN_VOLVALUE_USDT`

## 2. Çekirdek Pipeline
1. Sembol listesi alınır ve likiditeye göre filtrelenir.
2. Her sembol için eşzamanlı (Semaphore kısıtlı) OHLCV veri çekimi.
3. Aday üretim fonksiyonları: 
   - SMC (likidite süpürme → CHOCH + FVG/OTE)  
   - TREND (Donchian kırılımı + retest veya momentum)  
   - RANGE (Bollinger dar bant + false breakout re-enter bounce)  
   - MO (Momentum-breakout erken yakalama)  
4. Aday skorlaması: çoklu feature → ağırlıklandırılmış kompozit skor + logistic kalibrasyon.
5. Adaptif dinamik eşikler (`dyn_MIN_SCORE`) ve geçici gevşetme boş taramalarda.
6. Sinyal üretiminde ATR tabanlı SL ve R-multiples TP hesaplanır.
7. Telegram’a gönderim + takip (watcher) olayı.
8. Sinyal outcome evaluasyonu (bar ileri simülasyonu) ile online öğrenme (mini logit) + tuner.

## 3. Önemli Teknik Modüller
| Katman | Sorumluluk |
|--------|------------|
| Data Fetch | KuCoin REST kline & ticker çekimi |
| Indicators | ATR, ADX, RSI, Bollinger, Donchian, EMA, body strength |
| Feature Extraction | htf_align, adx_norm, ltf_momo, rr_norm, bw_adv, retest_or_fvg, atr_sweet, vol_pct, recent_penalty |
| Scoring | Ağırlıklar + base skor + TA kuralları ile final skor |
| Strategy | SMC / TREND / RANGE / MO birleşik seçici `pick_best_candidate` |
| Risk | ATR_STOP_MULT × ATR → SL, R-multiples → TP1/2/3 |
| Adaptation | Auto tuner + dyn_MIN_SCORE + boş taramada relax |
| AI Online | Logistic weights güncellemesi, p_final harmanlama |
| Follow Events | Donchian kırılım, BB re-enter, RSI zone, volume spike, ADX trend, swing bounce |

## 4. Sinyal Koşul Özeti
### TREND
- 1H ADX >= ADX_TREND_MIN & directional dispersion var
- Donchian üst/alt kırılım + (retest veya momentum_ok)

### RANGE
- ADX trend değil (zayıf trend) ve Bollinger bandwidth <= BWIDTH_RANGE
- Alt/üst band sarkması + re-enter + güçlü mum + hacim + RSI eşiği

### SMC
- Son swing yapısı: sweep → CHOCH (break of structure) + FVG (opsiyonel mod) + OTE fib zonu

### Momentum Breakout (MO)
- Donchian + EMA21 hizalı breakout
- Aşırı uzama / yüksek ATR filtresi
- Esnek momentum onayı (farklı modlar)

## 5. Skorlama Bileşenleri
```
Composite = SCORING_BASE + Σ (weight_i * feat_i) + kural düzeltmeleri
```
Örnek ağırlıklar (README): htf_align, adx_norm, ltf_momo, rr_norm(0), bw_adv, retest_or_fvg, atr_sweet, vol_pct, recent_penalty.

## 6. Adaptif & Online Öğrenme
- Sinyaller bar bazlı outcome ile TP/SL sınıflandırılır
- Logistic regresyon (mini) ağırlıkları güncellenir
- Otomatik tuner WR hedef sapmasına göre BASE_MIN_SCORE ve eşikleri iteratif ayarlar

## 7. Web Uygulaması Tasarım Hedefleri
| Hedef | Açıklama |
|-------|----------|
| Canlı Akış | WebSocket ile canlı sinyal push |
| Geçmiş | Son N sinyal tablo + skor grafiği |
| Analiz | Her sinyal için açıklama / feature detayı (ilerleyen faz) |
| Strateji Görselleme | Rejim (TREND/RANGE/SMC/MO) renk kodlama |
| İzleme | Takip edilen semboller için event log (ikinci faz) |
| Ayar Paneli | Mode switch, eşik güncelleme (ileri faz) |

## 8. Frontend Teknoloji Seçimi
- MVP: Mevcut FastAPI + Jinja + Vanilla + Chart.js (hızlı prototip) ✔
- Faz 2: Vite + React (durum yönetimi + component yapısı) → /frontend dizini.
- Neden iki aşama? Hızlı teslim + sonradan progressive enhancement.

## 9. Planlanan Genişletmeler
1. Ayrıntılı feature & explanation endpoint (`/api/signal/{id}`)
2. Event watcher verilerini websocket kanalına ekleme
3. Online tuner / ai ağırlıklarını gösteren panel
4. Auth (API key / basic) ve rate limit
5. Kalıcı depolama (SQLite / Postgres) → sinyal geçmişi + metrikler
6. Backtest endpoint (geriye dönük aday simülasyonu)
7. Dockerfile & CI pipeline

## 10. Deployment Önerisi
- Backend: Uvicorn + (opsiyonel) Nginx reverse proxy
- Süreklilik: systemd service veya Docker container + restart policy
- TELEGRAM_TOKEN environment olarak (.env / secret store)
- Izleme: Prometheus exporter (gelecek), basic health `/healthz`

## 11. Güvenlik Notları
- Token / API key asla repoda düz yazı tutulmamalı (README’deki token derhal iptal edilmeli)
- Rate-limit / retry backoff eklenmeli

## 12. Sonraki Adım
- Vite + React scaffold → WebSocket adaptör + sinyal store + ayarlar paneli.
- Bu doküman sürüm: v0.1
