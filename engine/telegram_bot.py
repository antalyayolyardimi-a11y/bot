"""Telegram entegrasyonu (opsiyonel)."""
from __future__ import annotations
import asyncio
from typing import Optional
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command

from .config import get_settings


class TelegramService:
    def __init__(self, engine):
        self.settings = get_settings()
        self.engine = engine
        self.bot: Optional[Bot] = None
        self.dp: Optional[Dispatcher] = None
        self._task: Optional[asyncio.Task] = None
        self._chat_id: Optional[int] = None

    def log(self, *a):
        print("[TG]", *a, flush=True)

    async def start(self):
        if not self.settings.TELEGRAM_TOKEN:
            self.log("TELEGRAM_TOKEN yok, telegram devre dışı.")
            return
        self.bot = Bot(token=self.settings.TELEGRAM_TOKEN)
        self.dp = Dispatcher()

        @self.dp.message(Command("start"))
        async def start_cmd(m: Message):
            self._chat_id = m.chat.id
            await m.answer("Bot aktif. /signals ile son sinyaller.")

        @self.dp.message(Command("signals"))
        async def signals_cmd(m: Message):
            lines = []
            for s in self.engine.get_live()[-10:]:
                lines.append(f"{s['symbol']} {s['side']} S={int(s['score'])} E={s['entry']:.6f} TP1={s['tp1']:.6f}")
            await m.answer("Son sinyaller:\n" + ("\n".join(lines) if lines else "-"))

        # Engine listener → anlık sinyal push
        def on_sig(sig):
            if self._chat_id and self.bot:
                text = (
                    f"🔔 {sig['symbol']} {sig['side']}\n"
                    f"Entry={sig['entry']:.6f} SL={sig['sl']:.6f} TP1={sig['tp1']:.6f}\n"
                    f"Skor={int(sig['score'])} | {sig['regime']} - {sig['reason']}"
                )
                asyncio.create_task(self.bot.send_message(self._chat_id, text))

        self.engine.add_listener(on_sig)
        self._task = asyncio.create_task(self.dp.start_polling(self.bot))

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
