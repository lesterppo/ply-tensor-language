"""
CMTIP Bus gRPC Server
═══════════════════════

Deploy the CMTIP Bus as a gRPC service for multi-agent tensor communication.

Usage:
    python cmtip_server.py [--port 50051]

Endpoints:
    - RegisterModel:    Register an embedding backend on the bus
    - SendText:         Send text, auto-embed, cross-project
    - SendTensor:       Send raw tensor directly
    - TrainAdapter:     Train cross-model W_{A→B} adapter
    - GetConversation:  Query conversation history
    - GetStatus:        Health/status check
    - Subscribe:        Bidirectional streaming for live conversations
"""

import grpc
import numpy as np
import json
import time
import struct
import argparse
import logging
from concurrent import futures
from collections import deque
from typing import Dict, List

import cmtip_service_pb2 as pb2
import cmtip_service_pb2_grpc as pb2_grpc

from cmtip import (
    CmtipBus, TensorPacket, SyntheticEmbeddingBackend,
    TensorDtype,
)

# Try real backends
try:
    from real_backends import SentenceTransformerBackend, OpenAIEmbeddingBackend
    HAS_ST = True
except ImportError:
    HAS_ST = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cmtip-server")


# ─── Packet Conversion ──────────────────────────────────────────────────────

def tensor_to_proto(packet: TensorPacket) -> pb2.TensorPacket:
    """Convert internal TensorPacket to protobuf."""
    return pb2.TensorPacket(
        source_id=packet.source_id,
        target_id=packet.target_id,
        seq_num=packet.seq_num,
        timestamp=packet.timestamp,
        tensor=packet.tensor.astype(np.float32).tobytes(),
        shape=list(packet.shape),
        dtype=int(packet.dtype),
        concept_tags=packet.concept_tags,
        confidence=packet.confidence,
        reply_to_seq=packet.reply_to_seq,
    )


def proto_to_tensor(proto: pb2.TensorPacket) -> TensorPacket:
    """Convert protobuf to internal TensorPacket."""
    tensor = np.frombuffer(proto.tensor, dtype=np.float32).reshape(tuple(proto.shape))
    return TensorPacket(
        source_id=proto.source_id,
        target_id=proto.target_id,
        seq_num=proto.seq_num,
        timestamp=proto.timestamp,
        tensor=tensor,
        shape=tuple(proto.shape),
        dtype=TensorDtype(proto.dtype),
        concept_tags=list(proto.concept_tags),
        confidence=proto.confidence,
        reply_to_seq=proto.reply_to_seq,
    )


# ─── gRPC Servicer ──────────────────────────────────────────────────────────

class CmtipBusServicer(pb2_grpc.CmtipBusServicer):
    """gRPC implementation wrapping the CmtipBus."""

    def __init__(self, adapters_dir: str = None):
        self.bus = CmtipBus()
        self.adapters_dir = adapters_dir or "./adapters"
        self.subscribers: List[grpc.ServicerContext] = []
        self._load_persisted_adapters()
        logger.info(f"CMTIP Bus server ready (penalty_threshold={self.bus.penalty_threshold})")

    def _load_persisted_adapters(self):
        """Load any previously saved adapters."""
        import os
        if os.path.isdir(self.adapters_dir):
            try:
                self.bus.load_adapters(self.adapters_dir)
                logger.info(f"Loaded {len(self.bus.adapters)} adapters from {self.adapters_dir}")
            except Exception as e:
                logger.warning(f"Could not load adapters: {e}")

    # ─── Control Plane ──────────────────────────────────────────────────

    def RegisterModel(self, request: pb2.RegisterModelRequest, context):
        """Register a model with its embedding backend."""
        backend_type = request.backend_type or "synthetic"

        if backend_type == "synthetic":
            backend = SyntheticEmbeddingBackend(
                model_id=request.model_id,
                dim=request.dim,
            )
        elif backend_type == "sentence_transformer":
            if not HAS_ST:
                return pb2.RegisterModelResponse(
                    success=False,
                    error="sentence-transformers not installed. pip install sentence-transformers",
                )
            config = json.loads(request.backend_config or "{}")
            model_name = config.get("model_name", "all-MiniLM-L6-v2")
            backend = SentenceTransformerBackend(request.model_id, model_name)
        elif backend_type == "openai":
            from real_backends import OpenAIEmbeddingBackend
            config = json.loads(request.backend_config or "{}")
            backend = OpenAIEmbeddingBackend(
                model_id=request.model_id,
                model_name=config.get("model_name", "text-embedding-3-small"),
            )
        else:
            return pb2.RegisterModelResponse(
                success=False,
                error=f"Unknown backend type: {backend_type}",
            )

        self.bus.register_model(backend)
        logger.info(f"Registered model: {request.model_id} (d={backend.dim}, type={backend_type})")

        return pb2.RegisterModelResponse(
            success=True,
            registered_dim=backend.dim,
        )

    def TrainAdapter(self, request: pb2.TrainAdapterRequest, context):
        """Train W_{source→target} adapter."""
        source_texts = list(request.source_texts)
        target_texts = list(request.target_texts)

        if len(source_texts) != len(target_texts):
            return pb2.TrainAdapterResponse(
                mse=-1.0,
                num_pairs=0,
                saved=False,
                adapter_path="",
            )

        paired = list(zip(source_texts, target_texts))
        mse = self.bus.train_adapter(request.source_id, request.target_id, paired)

        # Persist
        import os
        os.makedirs(self.adapters_dir, exist_ok=True)
        self.bus.save_adapters(self.adapters_dir)
        adapter_path = os.path.join(
            self.adapters_dir,
            f"{request.source_id}→{request.target_id}.npz"
        )

        logger.info(
            f"Trained adapter {request.source_id}→{request.target_id}: "
            f"MSE={mse:.6f}, {len(paired)} pairs"
        )

        return pb2.TrainAdapterResponse(
            mse=mse,
            num_pairs=len(paired),
            saved=True,
            adapter_path=adapter_path,
        )

    def GetConversation(self, request: pb2.GetConversationRequest, context):
        """Return conversation history."""
        limit = request.limit or 20
        packets = self.bus.conversations[-limit:]
        return pb2.GetConversationResponse(
            packets=[tensor_to_proto(p) for p in packets],
            total_packets=len(self.bus.conversations),
        )

    def GetStatus(self, request: pb2.StatusRequest, context):
        """Health check."""
        return pb2.StatusResponse(
            registered_models=len(self.bus.models),
            trained_adapters=len(self.bus.adapters),
            total_conversations=len(self.bus.conversations),
            model_ids=list(self.bus.models.keys()),
            penalty_threshold=self.bus.penalty_threshold,
        )

    # ─── Data Plane ─────────────────────────────────────────────────────

    def SendText(self, request: pb2.SendTextRequest, context):
        """Send text → auto-embed → cross-project → deliver."""
        try:
            packet = self.bus.send(
                source_id=request.source_id,
                target_id=request.target_id,
                text=request.text,
                concept_tags=list(request.concept_tags) if request.concept_tags else None,
            )
            entropy = self.bus.check_target_entropy(packet)
            penalty_triggered = entropy > self.bus.penalty_threshold

            self._notify_subscribers(packet)

            logger.debug(
                f"Send: {request.source_id}→{request.target_id} "
                f"seq={packet.seq_num}, entropy={entropy:.3f}"
            )

            return pb2.SendTextResponse(
                packet=tensor_to_proto(packet),
                penalty_triggered=penalty_triggered,
                target_entropy=entropy,
            )
        except KeyError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Model not registered: {e}")
            return pb2.SendTextResponse()

    def SendTensor(self, request: pb2.SendTensorRequest, context):
        """Send raw tensor."""
        try:
            tensor = np.frombuffer(request.tensor, dtype=np.float32).reshape(
                tuple(request.shape)
            )
            packet = self.bus.send(
                source_id=request.source_id,
                target_id=request.target_id,
                tensor=tensor,
                concept_tags=list(request.concept_tags) if request.concept_tags else None,
            )
            packet.confidence = request.confidence

            entropy = self.bus.check_target_entropy(packet)
            penalty_triggered = entropy > self.bus.penalty_threshold

            self._notify_subscribers(packet)

            return pb2.SendTensorResponse(
                packet=tensor_to_proto(packet),
                penalty_triggered=penalty_triggered,
                target_entropy=entropy,
            )
        except KeyError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Model not registered: {e}")
            return pb2.SendTensorResponse()

    # ─── Streaming ───────────────────────────────────────────────────────

    def Subscribe(self, request_iterator, context):
        """Bidirectional streaming: relay tensor packets between subscribers."""
        self.subscribers.append(context)
        logger.info(f"New subscriber connected (total: {len(self.subscribers)})")

        try:
            for proto_pkt in request_iterator:
                # Validate: packet must have at minimum a source_id
                if not proto_pkt.source_id:
                    continue

                packet = proto_to_tensor(proto_pkt)
                self.bus.conversations.append(packet)

                # Broadcast to all other subscribers
                dead = []
                for sub in self.subscribers:
                    if sub != context:
                        try:
                            if sub.is_active():
                                sub.write(tensor_to_proto(packet))
                            else:
                                dead.append(sub)
                        except Exception:
                            dead.append(sub)
                for d in dead:
                    if d in self.subscribers:
                        self.subscribers.remove(d)

                # Echo back to this subscriber
                yield tensor_to_proto(packet)

        except grpc.RpcError:
            pass
        finally:
            if context in self.subscribers:
                self.subscribers.remove(context)
            logger.info(f"Subscriber disconnected (total: {len(self.subscribers)})")

    def _notify_subscribers(self, packet: TensorPacket):
        """Push a new packet to all streaming subscribers."""
        proto = tensor_to_proto(packet)
        dead = []
        for sub in self.subscribers:
            if sub.is_active():
                try:
                    sub.write(proto)
                except Exception:
                    dead.append(sub)
            else:
                dead.append(sub)
        for d in dead:
            if d in self.subscribers:
                self.subscribers.remove(d)


# ─── Server Entrypoint ──────────────────────────────────────────────────────

def serve(port: int = 50051, max_workers: int = 10, adapters_dir: str = None):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),   # 50 MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ],
    )
    pb2_grpc.add_CmtipBusServicer_to_server(
        CmtipBusServicer(adapters_dir=adapters_dir), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"CMTIP Bus gRPC server listening on port {port}")
    logger.info(f"  Control plane:  gRPC :{port}")
    logger.info(f"  Data plane:     gRPC :{port} (streaming via Subscribe)")
    logger.info(f"  Adapters dir:   {adapters_dir or './adapters'}")
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMTIP Bus gRPC Server")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port (default: 50051)")
    parser.add_argument("--adapters-dir", type=str, default="./adapters",
                        help="Directory for persisted adapters")
    args = parser.parse_args()

    server = serve(port=args.port, adapters_dir=args.adapters_dir)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(0)
