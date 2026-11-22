from datetime import datetime, timedelta, timezone
from uuid import uuid4
from .schema import ns_episodic, ns_semantic, now_iso

MAX_EVENTS_PER_USER = 200

class MemoryRepo:
    """Public API backed by Langgraph store"""

    def __init__(self, store, episodic_ttl_days: int = 90, has_index: bool = True):
        self.store, self.ttl, self.has_index = store, episodic_ttl_days, has_index

    async def setup(self):
        await self.purge_all_expired_hint()

    # ---------- semantic: slot upsert, never embedded ----------
    async def get_profile(self, uid, allowed: set[str]) -> dict[str, str]:
        items = await self.store.asearch(ns_semantic(uid), limit = 50)
        return {i.key: i.value["value"] for i in items if i.key in allowed}

    async def upsert_slot(self, uid, slot, value, confidence, source = "explicit"):
        ns, t = ns_semantic(uid), now_iso()
        existing = await self.store.aget(ns, slot)
        await self.store.aput(ns, slot, {
            "slot":slot, "value": value, "source": source, "confidence": confidence,
            "created_at": existing.value["created_at"] if existing else t,
            "last_confirmed_at": t,
        }, index = False)

    # ---------- episodic ----------
    async def add_event(self, uid, event_type, text):
        expires = datetime.now(timezone.utc) + timedelta(days=self.ttl)
        await self.store.aput(ns_episodic(uid), uuid4().hex, {
            "type": event_type, "text": text,
            "created_at": now_iso(), "expires_at": expires.isoformat(),
        }, index=["text"])
        await self._cap(uid)

    async def _cap(self, uid):
        """Keep the newest MAX_EVENTS_PER_USER so 'recent' stays correct (Store has no ORDER BY)."""
        items = await self.store.asearch(ns_episodic(uid), limit=MAX_EVENTS_PER_USER + 50)
        if len(items) <= MAX_EVENTS_PER_USER:
            return
        items.sort(key=lambda i: i.value["created_at"])
        for it in items[: len(items) - MAX_EVENTS_PER_USER]:
            await self.store.adelete(ns_episodic(uid), it.key)

    @staticmethod
    def _live(items):
        now = now_iso()
        return [i for i in items if i.value["expires_at"] > now]

    @staticmethod
    def _fmt(i): return f'{i.value["created_at"][:10]}: {i.value["text"]}'

    async def recent_events(self, uid, limit=3) -> list[str]:
        items = self._live(await self.store.asearch(ns_episodic(uid), limit=MAX_EVENTS_PER_USER + 50))
        items.sort(key=lambda i: i.value["created_at"], reverse=True)
        return [self._fmt(i) for i in items[:limit]]

    async def search_events(self, uid, query, k=3, min_score=0.35) -> list[str]:
        if not self.has_index:
            return await self.recent_events(uid, k)
        items = await self.store.asearch(ns_episodic(uid), query=query, limit=k * 3)
        items = [i for i in self._live(items) if i.score is None or i.score >= min_score]
        return [self._fmt(i) for i in items[:k]]

    # ---------- governance ----------
    async def delete_slot(self, uid, slot):
        await self.store.adelete(ns_semantic(uid), slot)

    async def delete_all(self, uid):
        for ns in (ns_semantic(uid), ns_episodic(uid)):
            while True:
                items = await self.store.asearch(ns, limit=500)
                if not items:
                    break
                for it in items:
                    await self.store.adelete(ns, it.key)

    async def purge_expired(self, uid):
        for it in await self.store.asearch(ns_episodic(uid), limit=MAX_EVENTS_PER_USER + 50):
            if it.value["expires_at"] <= now_iso():
                await self.store.adelete(ns_episodic(uid), it.key)

    async def inspect_all(self, uid, allowed: set[str]) -> dict:
        return {"preferences": await self.get_profile(uid, allowed),
                "recent_events": await self.recent_events(uid, 50)}

    