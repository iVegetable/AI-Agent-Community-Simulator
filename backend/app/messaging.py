import asyncio
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[int, list[WebSocket]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def connect(self, simulation_id: int, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[simulation_id].append(websocket)

    async def disconnect(self, simulation_id: int, websocket: WebSocket) -> None:
        async with self._lock:
            if simulation_id in self._connections and websocket in self._connections[simulation_id]:
                self._connections[simulation_id].remove(websocket)

    async def broadcast(self, simulation_id: int, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._connections.get(simulation_id, []))
        for ws in targets:
            await ws.send_json(payload)
