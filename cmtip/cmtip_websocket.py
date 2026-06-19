"""
CMTIP WebSocket Streaming Server
═════════════════════════════════

Real-time model-to-model tensor dialog over WebSocket.
Connects to the gRPC bus for backend operations and exposes
a WebSocket interface for live bidirectional tensor streaming.

Architecture:
    Model A ←→ WebSocket ←→ CMTIP Bus (gRPC) ←→ WebSocket ←→ Model B

Protocol (JSON over WebSocket):

  → Client sends:
    {
      "type": "register",      // Register as a model
      "model_id": "llama-3",
      "dim": 384,
      "backend": "synthetic"   // optional
    }

    {
      "type": "text",          // Send text (server embeds + projects)
      "target_id": "deepseek",
      "text": "Hello in tensor space",
      "tags": ["greeting"]
    }

    {
      "type": "tensor",        // Send raw tensor (already embedded)
      "target_id": "deepseek",
      "tensor_b64": "...",     // base64-encoded float32 bytes
      "shape": [384],
      "tags": ["concept"]
    }

    {
      "type": "subscribe"      // Subscribe to all packets on the bus
    }

    {
      "type": "status"         // Request bus status
    }

  ← Server sends:
    {
      "type": "packet",        // A tensor packet was routed
      "source_id": "...",
      "target_id": "...",
      "seq_num": 0,
      "tensor_b64": "...",     // base64 float32 bytes
      "shape": [384],
      "tags": [...],
      "confidence": 0.95
    }

    {
      "type": "penalty",       // Entropy exceeded threshold
      "target_entropy": 0.85,
      "source_id": "..."
    }

    {
      "type": "status",        // Bus status response
      ...
    }

    {
      "type": "error",
      "message": "..."
    }

Usage:
    python cmtip_websocket.py [--port 8765] [--grpc-host localhost:50051]
"""

import asyncio
import json
import base64
import struct
import logging
import argparse
import sys
import os

import numpy as np
import websockets
from websockets.asyncio.server import serve

import grpc
import cmtip_service_pb2 as pb2
import cmtip_service_pb2_grpc as pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cmtip-ws")


# ─── gRPC Bridge ─────────────────────────────────────────────────────────────

class GrpcBridge:
    """Bridge to the CMTIP gRPC bus for backend operations."""

    def __init__(self, grpc_host: str = "localhost:50051"):
        self.channel = grpc.insecure_channel(
            grpc_host,
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ],
        )
        self.stub = pb2_grpc.CmtipBusStub(self.channel)
        # Track which model each websocket client is registered as
        self.client_models: dict = {}  # websocket → model_id

    async def ensure_registered(self, websocket, model_id: str, dim: int,
                                 backend: str = "synthetic", config: dict = None):
        """Ensure this client's model is registered on the bus."""
        try:
            resp = self.stub.RegisterModel(pb2.RegisterModelRequest(
                model_id=model_id,
                dim=dim,
                backend_type=backend,
                backend_config=json.dumps(config or {}),
            ))
            if resp.success:
                self.client_models[websocket] = model_id
                logger.info(f"Model {model_id} registered (dim={resp.registered_dim})")
                return True, resp.registered_dim
            return False, resp.error
        except grpc.RpcError as e:
            return False, str(e)

    def send_text_sync(self, source_id: str, target_id: str, text: str,
                        tags: list = None):
        """Send text through the gRPC bus (synchronous)."""
        resp = self.stub.SendText(pb2.SendTextRequest(
            source_id=source_id,
            target_id=target_id,
            text=text,
            concept_tags=tags or [],
        ))
        return {
            "source_id": resp.packet.source_id,
            "target_id": resp.packet.target_id,
            "seq_num": resp.packet.seq_num,
            "tensor": resp.packet.tensor,
            "shape": list(resp.packet.shape),
            "tags": list(resp.packet.concept_tags),
            "confidence": resp.packet.confidence,
            "penalty_triggered": resp.penalty_triggered,
            "target_entropy": resp.target_entropy,
        }

    def send_tensor_sync(self, source_id: str, target_id: str,
                          tensor: np.ndarray, shape: list,
                          tags: list = None, confidence: float = 1.0):
        """Send raw tensor through the gRPC bus (synchronous)."""
        resp = self.stub.SendTensor(pb2.SendTensorRequest(
            source_id=source_id,
            target_id=target_id,
            tensor=tensor.astype(np.float32).tobytes(),
            shape=shape,
            concept_tags=tags or [],
            confidence=confidence,
        ))
        return {
            "source_id": resp.packet.source_id,
            "target_id": resp.packet.target_id,
            "seq_num": resp.packet.seq_num,
            "penalty_triggered": resp.penalty_triggered,
            "target_entropy": resp.target_entropy,
        }

    def get_status(self) -> dict:
        resp = self.stub.GetStatus(pb2.StatusRequest())
        return {
            "registered_models": resp.registered_models,
            "trained_adapters": resp.trained_adapters,
            "total_conversations": resp.total_conversations,
            "model_ids": list(resp.model_ids),
        }

    def close(self):
        self.channel.close()


# ─── WebSocket Handler ───────────────────────────────────────────────────────

class CmtipWebSocketServer:
    """WebSocket server for live tensor streaming."""

    def __init__(self, grpc_bridge: GrpcBridge):
        self.grpc = grpc_bridge
        # Connected clients (websocket objects)
        self.clients: set = set()
        # Clients subscribed to broadcast packets
        self.subscribers: set = set()

    async def handler(self, websocket):
        """Handle a single WebSocket connection."""
        self.clients.add(websocket)
        remote = websocket.remote_address
        logger.info(f"Client connected: {remote} (total: {len(self.clients)})")

        try:
            async for raw_message in websocket:
                try:
                    msg = json.loads(raw_message)
                    response = await self._dispatch(websocket, msg)
                    if response:
                        await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON",
                    }))
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e),
                    }))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            self.subscribers.discard(websocket)
            self.grpc.client_models.pop(websocket, None)
            logger.info(f"Client disconnected: {remote} (total: {len(self.clients)})")

    async def _dispatch(self, websocket, msg: dict) -> dict:
        """Dispatch a WebSocket message to the appropriate handler."""
        msg_type = msg.get("type", "")

        if msg_type == "register":
            return await self._handle_register(websocket, msg)

        elif msg_type == "text":
            return await self._handle_text(websocket, msg)

        elif msg_type == "tensor":
            return await self._handle_tensor(websocket, msg)

        elif msg_type == "subscribe":
            self.subscribers.add(websocket)
            return {"type": "subscribed", "message": "Receiving live packets"}

        elif msg_type == "unsubscribe":
            self.subscribers.discard(websocket)
            return {"type": "unsubscribed"}

        elif msg_type == "status":
            s = self.grpc.get_status()
            s["type"] = "status"
            return s

        else:
            return {"type": "error", "message": f"Unknown message type: {msg_type}"}

    async def _handle_register(self, websocket, msg: dict) -> dict:
        """Register a model."""
        model_id = msg["model_id"]
        dim = msg.get("dim", 384)
        backend = msg.get("backend", "synthetic")
        config = msg.get("config")

        success, result = await self.grpc.ensure_registered(
            websocket, model_id, dim, backend, config
        )
        if success:
            return {
                "type": "registered",
                "model_id": model_id,
                "dim": result,
            }
        return {"type": "error", "message": f"Registration failed: {result}"}

    async def _handle_text(self, websocket, msg: dict) -> dict:
        """Send text through the bus."""
        source_id = self.grpc.client_models.get(websocket)
        if not source_id:
            return {"type": "error", "message": "Not registered. Send 'register' first."}

        target_id = msg["target_id"]
        text = msg["text"]
        tags = msg.get("tags", [])

        # Use asyncio to run gRPC call without blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.grpc.send_text_sync(source_id, target_id, text, tags)
        )

        # Broadcast to subscribers
        broadcast_msg = {
            "type": "packet",
            "source_id": result["source_id"],
            "target_id": result["target_id"],
            "seq_num": result["seq_num"],
            "tensor_b64": base64.b64encode(result["tensor"]).decode(),
            "shape": result["shape"],
            "tags": result["tags"],
            "confidence": result["confidence"],
        }
        await self._broadcast(broadcast_msg, exclude=websocket)

        if result["penalty_triggered"]:
            await self._broadcast({
                "type": "penalty",
                "source_id": result["source_id"],
                "target_entropy": result["target_entropy"],
            })

        return {"type": "sent", "seq_num": result["seq_num"],
                "penalty_triggered": result["penalty_triggered"]}

    async def _handle_tensor(self, websocket, msg: dict) -> dict:
        """Send raw tensor through the bus."""
        source_id = self.grpc.client_models.get(websocket)
        if not source_id:
            return {"type": "error", "message": "Not registered. Send 'register' first."}

        target_id = msg["target_id"]
        tensor_bytes = base64.b64decode(msg["tensor_b64"])
        shape = msg["shape"]
        tensor = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(shape)
        tags = msg.get("tags", [])
        confidence = msg.get("confidence", 1.0)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.grpc.send_tensor_sync(
                source_id, target_id, tensor, shape, tags, confidence
            )
        )

        return {"type": "sent", "seq_num": result["seq_num"],
                "penalty_triggered": result["penalty_triggered"]}

    async def _broadcast(self, message: dict, exclude=None):
        """Broadcast a message to all subscribers."""
        dead = set()
        for client in self.subscribers:
            if client == exclude:
                continue
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                dead.add(client)
            except Exception:
                dead.add(client)
        self.subscribers -= dead
        self.clients -= dead


# ─── Entrypoint ──────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="CMTIP WebSocket Streaming Server")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--grpc-host", default="localhost:50051",
                        help="gRPC bus host:port")
    args = parser.parse_args()

    grpc_bridge = GrpcBridge(args.grpc_host)
    ws_server = CmtipWebSocketServer(grpc_bridge)

    logger.info(f"CMTIP WebSocket server starting on ws://0.0.0.0:{args.port}")
    logger.info(f"  gRPC backend: {args.grpc_host}")
    logger.info(f"  Protocol: JSON over WebSocket")
    logger.info(f"  Connect:   ws://localhost:{args.port}")

    try:
        async with serve(ws_server.handler, "0.0.0.0", args.port):
            await asyncio.get_running_loop().create_future()  # run forever
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        grpc_bridge.close()


if __name__ == "__main__":
    asyncio.run(main())
