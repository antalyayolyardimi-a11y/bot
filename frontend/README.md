# Frontend (Planlanan Faz 2)

Bu dizin ilerleyen adımda Vite + React tabanlı modern SPA arayüzü için kullanılacaktır.

## Hedef Özellikler
- WebSocket canlı sinyal feed tüketimi
- Tablo + filtre + arama
- Sinyal detay modal (feature + explanation)
- Event watcher pane (takip edilen semboller)
- Ayar paneli (eşikler, mode switch)
- Dark / Light theme toggle

## Başlangıç Scaffold Komutları (oluşturulmadı henüz)
```
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm run dev
```

Backende bağlanmak için env:
```
VITE_API_BASE=http://localhost:8000
```

WebSocket endpoint: `ws://localhost:8000/ws/signals`

Bu README, faz 1 (Jinja prototip) stabilize olduktan sonra güncellenecektir.
