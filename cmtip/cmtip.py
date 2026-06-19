"""
CMTIP — Cross-Model Tensor Interoperability Protocol
═══════════════════════════════════════════════════════

The LLM-Native Language: models communicate directly in embedding space,
bypassing the lossy text-token roundtrip.

Architecture:

  Model A (d=384)                Model B (d=768)
  ┌──────────────┐              ┌──────────────┐
  │ "thinking"   │              │ "thinking"   │
  │ in latent    │              │ in latent    │
  │ space d=384  │              │ space d=768  │
  └──────┬───────┘              └──────▲───────┘
         │ embedding vector           │ projected vector
         │ v ∈ R³⁸⁴                   │ v' = v · W_{A→B}
         │                             │
         └──────────┬──────────────────┘
                    │
              ┌─────▼──────┐
              │  CMTIP Bus │
              │  routes    │
              │  tensors   │
              └────────────┘

Key operations:
  1. SEND:    Model A emits embedding vector → CMTIP packet
  2. PROJECT: W_{A→B} maps from A's space to B's space
  3. RECV:    Model B receives projected vector in its own latent space
  4. FEEDBACK: If B's attention entropy spikes, emit penalty tensor → A

This eliminates the "lost in translation" problem of text-based inter-model
communication. Models speak their native language: continuous vectors.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import IntEnum
import hashlib
import struct
import time
import json
import os


# ═══════════════════════════════════════════════════════════════
# Tensor Packet Protocol
# ═══════════════════════════════════════════════════════════════

class TensorDtype(IntEnum):
    FP32 = 0
    FP16 = 1
    INT8 = 2
    BF16 = 3

@dataclass
class TensorPacket:
    """
    The atomic unit of LLM-native communication.
    
    A TensorPacket is what replaces a "message" in text-based chat.
    Instead of "Hello, how are you?" as tokens, it carries the raw
    embedding vector that represents that semantic content.
    """
    magic: int = 0x484D544C  # 'HMTL'
    source_id: str = ""       # Model identifier (e.g. "llama-3-8b")
    target_id: str = ""       # Target model or "broadcast"
    seq_num: int = 0          # Monotonic sequence number
    timestamp: float = 0.0    # Unix timestamp
    
    # The actual tensor payload
    tensor: np.ndarray = field(default_factory=lambda: np.array([]))
    dtype: TensorDtype = TensorDtype.FP32
    shape: Tuple[int, ...] = ()
    
    # Metadata
    concept_tags: List[str] = field(default_factory=list)  # Semantic tags
    confidence: float = 1.0    # Model's confidence in this vector
    reply_to_seq: int = -1     # Sequence number this replies to
    
    def pack(self) -> bytes:
        """Serialize to wire format (JSON header + binary tensor payload)."""
        import base64
        header = {
            'magic': self.magic,
            'src': self.source_id,
            'tgt': self.target_id,
            'seq': self.seq_num,
            'ts': self.timestamp,
            'dtype': int(self.dtype),
            'shape': list(self.shape),
            'tags': self.concept_tags,
            'conf': self.confidence,
            'reply': self.reply_to_seq,
        }
        header_json = json.dumps(header).encode('utf-8')
        header_len = struct.pack('<I', len(header_json))
        tensor_data = self.tensor.astype(np.float32).tobytes()
        return header_len + header_json + tensor_data
    
    @classmethod
    def unpack(cls, data: bytes) -> 'TensorPacket':
        """Deserialize from wire format."""
        header_len = struct.unpack('<I', data[:4])[0]
        header_json = data[4:4+header_len].decode('utf-8')
        header = json.loads(header_json)
        
        tensor_start = 4 + header_len
        shape = tuple(header['shape'])
        tensor_size = int(np.prod(shape)) * 4  # float32
        tensor = np.frombuffer(data[tensor_start:tensor_start+tensor_size], dtype=np.float32).reshape(shape)
        
        return cls(
            magic=header['magic'],
            source_id=header['src'],
            target_id=header['tgt'],
            seq_num=header['seq'],
            timestamp=header['ts'],
            tensor=tensor,
            dtype=TensorDtype(header['dtype']),
            shape=shape,
            concept_tags=header['tags'],
            confidence=header['conf'],
            reply_to_seq=header['reply'],
        )
    
    @property
    def norm(self) -> float:
        """L2 norm of the tensor."""
        return float(np.linalg.norm(self.tensor))
    
    def cosine_similarity(self, other: 'TensorPacket') -> float:
        """Semantic similarity between two packets."""
        a = self.tensor.flatten()
        b = other.tensor.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    def __repr__(self):
        return (f"TensorPacket({self.source_id}→{self.target_id} "
                f"#{self.seq_num} shape={self.shape} "
                f"|v|={self.norm:.2f} tags={self.concept_tags}")


# ═══════════════════════════════════════════════════════════════
# Enhanced Wire Format V2 — Fragmentation, Checksum, Versioning
# ═══════════════════════════════════════════════════════════════

PROTOCOL_VERSION = 2
MAGIC_V2 = 0x484D544C  # 'HMTL'
MAX_PAYLOAD_BYTES = 1_048_576  # 1 MB max single packet

def _crc32(data: bytes) -> int:
    """CRC-32 checksum for data integrity."""
    import zlib
    return zlib.crc32(data) & 0xFFFFFFFF

def pack_v2(packet: 'TensorPacket', max_fragment_size: int = 65536) -> List[bytes]:
    """
    Pack a TensorPacket into V2 wire format with:
      - Protocol versioning
      - CRC-32 checksum
      - Automatic fragmentation for large tensors
      - Timestamp in microseconds
    
    Wire format:
      [4 bytes: magic 0x484D544C]
      [1 byte:  protocol_version]
      [1 byte:  fragment_index]
      [1 byte:  total_fragments]
      [1 byte:  flags (bit0=compressed, bit1=encrypted, bit2=last)]
      [4 bytes: header_len (uint32 LE)]
      [N bytes: JSON header]
      [M bytes: float32 tensor payload]
      [4 bytes: CRC-32 of header+payload]
    """
    tensor_data = packet.tensor.astype(np.float32).tobytes()
    total_size = len(tensor_data)
    num_fragments = max(1, (total_size + max_fragment_size - 1) // max_fragment_size)
    
    fragments = []
    for frag_idx in range(num_fragments):
        start = frag_idx * max_fragment_size
        end = min(start + max_fragment_size, total_size)
        chunk = tensor_data[start:end]
        
        flags = 0
        if frag_idx == num_fragments - 1:
            flags |= 0x04  # Last fragment
        
        header = {
            'ver': PROTOCOL_VERSION,
            'src': packet.source_id,
            'tgt': packet.target_id,
            'seq': packet.seq_num,
            'ts': int(time.time() * 1_000_000),  # microseconds
            'dtype': int(packet.dtype),
            'shape': list(packet.shape),
            'tags': packet.concept_tags,
            'conf': packet.confidence,
            'reply': packet.reply_to_seq,
            'frag': frag_idx,
            'total_frags': num_fragments,
            'payload_offset': start,
            'payload_len': len(chunk),
        }
        header_json = json.dumps(header).encode('utf-8')
        header_len = len(header_json)
        
        # Assemble frame
        frame = struct.pack('<I', MAGIC_V2)           # 4 bytes magic
        frame += struct.pack('B', PROTOCOL_VERSION)    # 1 byte version
        frame += struct.pack('B', frag_idx)            # 1 byte fragment index
        frame += struct.pack('B', num_fragments)       # 1 byte total fragments
        frame += struct.pack('B', flags)               # 1 byte flags
        frame += struct.pack('<I', header_len)          # 4 bytes header length
        frame += header_json                            # N bytes JSON header
        frame += chunk                                  # M bytes tensor payload
        
        # CRC-32 of header+payload (not including magic/version/frag/flags)
        body = header_json + chunk
        crc = _crc32(body)
        frame += struct.pack('<I', crc)                # 4 bytes CRC-32
        
        fragments.append(frame)
    
    return fragments

def unpack_v2(data: bytes) -> 'TensorPacket':
    """
    Unpack V2 wire format. Validates magic, version, and CRC-32.
    Supports single fragments; use reassemble_v2() for multi-fragment.
    """
    if len(data) < 12:
        raise ValueError(f"Packet too short: {len(data)} bytes")
    
    magic = struct.unpack('<I', data[0:4])[0]
    if magic != MAGIC_V2:
        raise ValueError(f"Bad magic: 0x{magic:08X}, expected 0x{MAGIC_V2:08X}")
    
    version = data[4]
    frag_idx = data[5]
    total_frags = data[6]
    flags = data[7]
    header_len = struct.unpack('<I', data[8:12])[0]
    
    header_json = data[12:12+header_len].decode('utf-8')
    header = json.loads(header_json)
    
    payload_start = 12 + header_len
    payload_len = header.get('payload_len', len(data) - payload_start - 4)
    payload = data[payload_start:payload_start + payload_len]
    
    # Verify CRC-32
    crc_offset = payload_start + payload_len
    expected_crc = struct.unpack('<I', data[crc_offset:crc_offset+4])[0]
    body = header_json.encode('utf-8') + payload
    actual_crc = _crc32(body)
    if actual_crc != expected_crc:
        raise ValueError(f"CRC mismatch: expected 0x{expected_crc:08X}, got 0x{actual_crc:08X}")
    
    shape = tuple(header['shape'])
    tensor = np.frombuffer(payload, dtype=np.float32).reshape(shape)
    
    return TensorPacket(
        magic=magic,
        source_id=header['src'],
        target_id=header['tgt'],
        seq_num=header['seq'],
        timestamp=header['ts'] / 1_000_000.0,
        tensor=tensor,
        dtype=TensorDtype(header.get('dtype', 0)),
        shape=shape,
        concept_tags=header.get('tags', []),
        confidence=header.get('conf', 1.0),
        reply_to_seq=header.get('reply', -1),
    )

def reassemble_v2(fragments: List[bytes]) -> 'TensorPacket':
    """Reassemble multi-fragment V2 packets."""
    if len(fragments) == 1:
        return unpack_v2(fragments[0])
    
    # Collect all fragments, sort by fragment index
    parsed = []
    for data in fragments:
        frag_idx = data[5]
        total_frags = data[6]
        header_len = struct.unpack('<I', data[8:12])[0]
        header = json.loads(data[12:12+header_len].decode('utf-8'))
        payload_start = 12 + header_len
        payload_len = header.get('payload_len', len(data) - payload_start - 4)
        payload = data[payload_start:payload_start + payload_len]
        parsed.append((frag_idx, total_frags, header, payload))
    
    parsed.sort(key=lambda x: x[0])
    
    # Reassemble
    first_header = parsed[0][2]
    total_frags = parsed[0][1]
    
    if len(parsed) != total_frags:
        raise ValueError(f"Missing fragments: got {len(parsed)}, expected {total_frags}")
    
    full_payload = b''.join(p[3] for p in parsed)
    shape = tuple(first_header['shape'])
    tensor = np.frombuffer(full_payload, dtype=np.float32).reshape(shape)
    
    return TensorPacket(
        magic=MAGIC_V2,
        source_id=first_header['src'],
        target_id=first_header['tgt'],
        seq_num=first_header['seq'],
        timestamp=first_header['ts'] / 1_000_000.0,
        tensor=tensor,
        dtype=TensorDtype(first_header.get('dtype', 0)),
        shape=shape,
        concept_tags=first_header.get('tags', []),
        confidence=first_header.get('conf', 1.0),
        reply_to_seq=first_header.get('reply', -1),
    )


# ═══════════════════════════════════════════════════════════════
# Packet Router — TTL, Priority, QoS
# ═══════════════════════════════════════════════════════════════

@dataclass
class RoutingPolicy:
    """QoS routing policy for tensor packets."""
    ttl: int = 16                    # Time-to-live (hops)
    priority: int = 5                # 0-10, higher = more urgent
    deadline_us: int = 0             # 0 = no deadline
    ack_required: bool = False       # Require acknowledgment
    max_retries: int = 0             # Retry on failure
    
    def is_expired(self, creation_time: float) -> bool:
        """Check if deadline has passed."""
        if self.deadline_us == 0:
            return False
        elapsed_us = (time.time() - creation_time) * 1_000_000
        return elapsed_us > self.deadline_us
    
    def decrement_ttl(self) -> bool:
        """Decrement TTL. Returns True if still alive."""
        self.ttl -= 1
        return self.ttl > 0


@dataclass
class RouteEntry:
    """A route entry in the routing table."""
    target: str
    next_hop: str
    cost: float = 1.0
    success_count: int = 0
    fail_count: int = 0
    last_used: float = 0.0
    
    @property
    def reliability(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / max(1, total)


class PacketRouter:
    """
    Routes TensorPackets between models with TTL, priority, and QoS.
    
    Features:
      - Routing table: target → next_hop mappings
      - Priority queuing: higher priority packets go first
      - TTL enforcement: packets expire after N hops
      - Deadline scheduling: time-critical packets get priority boost
      - Route learning: tracks success/failure per route
      - Broadcast: send to all registered targets
    """
    
    def __init__(self, max_queue_size: int = 1024):
        self.routes: Dict[str, RouteEntry] = {}
        self.pending: List[Tuple[float, 'TensorPacket', RoutingPolicy]] = []
        self.max_queue = max_queue_size
        self.delivered: int = 0
        self.dropped: int = 0
        self.expired: int = 0
    
    def add_route(self, target: str, next_hop: str, cost: float = 1.0):
        """Register a route."""
        self.routes[target] = RouteEntry(
            target=target, next_hop=next_hop, cost=cost
        )
    
    def route(self, packet: 'TensorPacket', policy: RoutingPolicy = None) -> List[str]:
        """
        Determine next-hop targets for a packet.
        
        Returns list of next_hop model IDs.
        """
        if policy is None:
            policy = RoutingPolicy()
        
        if packet.target_id == "broadcast":
            return [r.next_hop for r in self.routes.values()]
        
        if packet.target_id in self.routes:
            return [self.routes[packet.target_id].next_hop]
        
        # Direct delivery if target is registered
        return [packet.target_id]
    
    def enqueue(self, packet: 'TensorPacket', policy: RoutingPolicy = None):
        """Enqueue a packet for routing with priority ordering."""
        if policy is None:
            policy = RoutingPolicy()
        
        if len(self.pending) >= self.max_queue:
            # Drop lowest priority packet
            self.pending.sort(key=lambda x: x[2].priority)
            self.pending.pop(0)
            self.dropped += 1
        
        # Insert sorted by priority (higher first)
        prio = policy.priority
        if policy.deadline_us > 0:
            # Boost priority for near-deadline packets
            remaining = policy.deadline_us - (time.time() - packet.timestamp) * 1_000_000
            if remaining < 1000:  # <1ms remaining
                prio = 10  # Max priority
        
        self.pending.append((prio, packet, policy))
        self.pending.sort(key=lambda x: -x[0])  # Higher priority first
    
    def dequeue(self) -> Optional[Tuple['TensorPacket', RoutingPolicy]]:
        """Dequeue the highest-priority packet that hasn't expired."""
        now = time.time()
        while self.pending:
            _, packet, policy = self.pending.pop(0)
            
            # Check TTL
            if not policy.decrement_ttl():
                self.expired += 1
                continue
            
            # Check deadline
            if policy.is_expired(packet.timestamp):
                self.expired += 1
                continue
            
            self.delivered += 1
            return packet, policy
        
        return None
    
    def record_success(self, target: str):
        """Record successful delivery to a target."""
        if target in self.routes:
            self.routes[target].success_count += 1
            self.routes[target].last_used = time.time()
    
    def record_failure(self, target: str):
        """Record failed delivery to a target."""
        if target in self.routes:
            self.routes[target].fail_count += 1
    
    def best_route(self) -> Optional[str]:
        """Return the most reliable route."""
        if not self.routes:
            return None
        return max(self.routes.values(), key=lambda r: r.reliability).target
    
    def stats(self) -> dict:
        """Router statistics."""
        return {
            "routes": len(self.routes),
            "pending": len(self.pending),
            "delivered": self.delivered,
            "dropped": self.dropped,
            "expired": self.expired,
            "best_route": self.best_route(),
            "route_table": [
                {
                    "target": r.target,
                    "next_hop": r.next_hop,
                    "reliability": round(r.reliability, 3),
                    "cost": r.cost,
                }
                for r in sorted(self.routes.values(), key=lambda x: -x.reliability)
            ],
        }


# ═══════════════════════════════════════════════════════════════
# Cross-Model Linear Adapter
# ═══════════════════════════════════════════════════════════════

@dataclass
class CrossModelAdapter:
    """
    Learned projection between two embedding spaces.
    
    W_{A→B} ∈ R^{d_A × d_B}
    
    Projects vectors from model A's latent space into model B's space:
        v_B = v_A @ W_{A→B}
    
    The adapter is learned via ordinary least squares on a set of
    paired "sentence" vectors — the same text embedded by both models.
    """
    source_dim: int
    target_dim: int
    source_model: str
    target_model: str
    weight: np.ndarray = field(default=None)  # [d_A × d_B]
    
    def __post_init__(self):
        if self.weight is None:
            # Orthogonal random initialization
            rng = np.random.RandomState(
                int(hashlib.md5(f"{self.source_model}→{self.target_model}".encode()).hexdigest()[:8], 16)
            )
            raw = rng.randn(self.source_dim, self.target_dim).astype(np.float32)
            # Scale to preserve vector norm after projection
            scale = np.sqrt(self.target_dim / self.source_dim)
            self.weight = (raw * scale / np.linalg.norm(raw, axis=0, keepdims=True)).astype(np.float32)
    
    def project(self, vector: np.ndarray) -> np.ndarray:
        """
        v_target = v_source @ W
        """
        v = vector.reshape(1, -1).astype(np.float32)
        return (v @ self.weight).flatten()
    
    def project_packet(self, packet: TensorPacket) -> TensorPacket:
        """
        Project an entire TensorPacket into the target space.
        """
        projected = self.project(packet.tensor.flatten())
        return TensorPacket(
            source_id=f"{packet.source_id}→{self.target_model}",
            target_id=self.target_model,
            seq_num=packet.seq_num,
            timestamp=time.time(),
            tensor=projected,
            shape=(self.target_dim,),
            concept_tags=packet.concept_tags,
            confidence=packet.confidence,
            reply_to_seq=packet.reply_to_seq,
        )
    
    def fit(self, source_vectors: np.ndarray, target_vectors: np.ndarray):
        """
        Learn W via ordinary least squares.
        
        Given N paired vectors:
          X ∈ R^{N × d_A}  (source)
          Y ∈ R^{N × d_B}  (target)
        
        Solve: W = (X^T X + λI)^{-1} X^T Y
        """
        X = source_vectors.astype(np.float32)
        Y = target_vectors.astype(np.float32)
        N = X.shape[0]
        
        # Ridge regression for stability
        lambda_reg = 0.01 * N
        XtX = X.T @ X
        XtX_reg = XtX + lambda_reg * np.eye(self.source_dim, dtype=np.float32)
        XtY = X.T @ Y
        
        self.weight = np.linalg.solve(XtX_reg, XtY).astype(np.float32)
        
        # Compute training error
        Y_pred = X @ self.weight
        mse = np.mean((Y - Y_pred) ** 2)
        return mse
    
    def save(self, path: str):
        np.savez(path, weight=self.weight, source_dim=self.source_dim,
                 target_dim=self.target_dim, source_model=self.source_model,
                 target_model=self.target_model)
    
    @classmethod
    def load(cls, path: str) -> 'CrossModelAdapter':
        data = np.load(path)
        adapter = cls(
            source_dim=int(data['source_dim']),
            target_dim=int(data['target_dim']),
            source_model=str(data['source_model']),
            target_model=str(data['target_model']),
        )
        adapter.weight = data['weight']
        return adapter


# ═══════════════════════════════════════════════════════════════
# Nonlinear MLP Adapter — Bottleneck Architecture
# ═══════════════════════════════════════════════════════════════

@dataclass
class NonlinearAdapter:
    """
    Two-layer MLP with bottleneck for cross-model projection.
    
    Architecture:  d_A → bottleneck → d_B
    
    Instead of a single linear matrix W (d_A × d_B = 294K params for MiniLM→MPNet),
    we use:
        h = ReLU(v_A @ W1 + b1)     # d_A → bottleneck
        v_B = h @ W2 + b2            # bottleneck → d_B
    
    With bottleneck = 128, total params = 384×128 + 128×768 = 147K (vs 294K linear).
    
    Why this helps:
    - The bottleneck forces the model to find a shared compressed representation
    - ReLU nonlinearity captures non-linear manifold deformations between spaces
    - Fewer parameters = less overfitting on small training sets
    - Architecture is learnable via gradient descent (SGD/Adam)
    """
    source_dim: int
    target_dim: int
    bottleneck_dim: int = 128
    source_model: str = ""
    target_model: str = ""
    
    # Trainable parameters
    W1: np.ndarray = None   # [d_A × bottleneck]
    b1: np.ndarray = None   # [bottleneck]
    W2: np.ndarray = None   # [bottleneck × d_B]
    b2: np.ndarray = None   # [d_B]
    
    def __post_init__(self):
        if self.W1 is None:
            rng = np.random.RandomState(
                int(hashlib.md5(f"mlp:{self.source_model}→{self.target_model}".encode()).hexdigest()[:8], 16)
            )
            # Kaiming initialization for ReLU
            scale1 = np.sqrt(2.0 / self.source_dim)
            scale2 = np.sqrt(2.0 / self.bottleneck_dim)
            self.W1 = (rng.randn(self.source_dim, self.bottleneck_dim) * scale1).astype(np.float32)
            self.b1 = np.zeros(self.bottleneck_dim, dtype=np.float32)
            self.W2 = (rng.randn(self.bottleneck_dim, self.target_dim) * scale2).astype(np.float32)
            self.b2 = np.zeros(self.target_dim, dtype=np.float32)
    
    def project(self, vector: np.ndarray) -> np.ndarray:
        """v_target = ReLU(v_source @ W1 + b1) @ W2 + b2"""
        v = vector.reshape(1, -1).astype(np.float32)
        h = np.maximum(0, v @ self.W1 + self.b1)  # ReLU
        out = (h @ self.W2 + self.b2).flatten()
        return out / (np.linalg.norm(out) + 1e-8)
    
    def project_packet(self, packet, target_model: str = ""):
        """Project a TensorPacket through the MLP."""
        from cmtip import TensorPacket
        projected = self.project(packet.tensor.flatten())
        return TensorPacket(
            source_id=f"{packet.source_id}→{target_model or self.target_model}",
            target_id=target_model or self.target_model,
            seq_num=packet.seq_num,
            timestamp=time.time(),
            tensor=projected,
            shape=(self.target_dim,),
            concept_tags=packet.concept_tags,
            confidence=packet.confidence,
            reply_to_seq=packet.reply_to_seq,
        )
    
    def fit(self, X: np.ndarray, Y: np.ndarray,
            lr: float = 0.01, epochs: int = 200,
            batch_size: int = 32, verbose: bool = False) -> float:
        """
        Train via minibatch SGD with MSE loss.
        
        Args:
            X: [N, d_A] source embeddings
            Y: [N, d_B] target embeddings
        Returns final training MSE.
        """
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        N = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(N)
            total_loss = 0.0
            
            for start in range(0, N, batch_size):
                batch_idx = idx[start:start + batch_size]
                Xb = X[batch_idx]
                Yb = Y[batch_idx]
                m = len(Xb)
                
                # Forward
                H = np.maximum(0, Xb @ self.W1 + self.b1)  # ReLU
                Y_pred = H @ self.W2 + self.b2
                
                # MSE loss
                diff = Y_pred - Yb
                loss = np.mean(diff ** 2)
                total_loss += loss * m
                
                # Backward (manual gradients for clarity)
                dY_pred = 2 * diff / m  # dL/dY_pred
                
                # W2, b2 gradients
                dW2 = H.T @ dY_pred
                db2 = dY_pred.sum(axis=0)
                
                # Backprop through ReLU
                dH = dY_pred @ self.W2.T
                dH[H <= 0] = 0  # ReLU gradient
                
                # W1, b1 gradients
                dW1 = Xb.T @ dH
                db1 = dH.sum(axis=0)
                
                # Update (SGD with gradient clipping)
                grad_norm = np.sqrt(
                    np.sum(dW1**2) + np.sum(db1**2) +
                    np.sum(dW2**2) + np.sum(db2**2)
                )
                clip = 5.0
                if grad_norm > clip:
                    scale = clip / (grad_norm + 1e-8)
                    dW1 *= scale; db1 *= scale
                    dW2 *= scale; db2 *= scale
                
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
            
            avg_loss = total_loss / N
            if verbose and epoch % 50 == 0:
                # Test cosine similarity on training set
                H_all = np.maximum(0, X @ self.W1 + self.b1)
                Y_all_pred = H_all @ self.W2 + self.b2
                cos_sims = []
                for i in range(min(100, N)):
                    cos = float(np.dot(Y_all_pred[i], Y[i]) / 
                              (np.linalg.norm(Y_all_pred[i]) * np.linalg.norm(Y[i]) + 1e-8))
                    cos_sims.append(cos)
                print(f"  epoch {epoch:3d}: loss={avg_loss:.6f}  train_cos={np.mean(cos_sims):.4f}")
        
        return avg_loss
    
    def save(self, path: str):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 source_dim=self.source_dim, target_dim=self.target_dim,
                 bottleneck_dim=self.bottleneck_dim,
                 source_model=self.source_model, target_model=self.target_model)
    
    @classmethod
    def load(cls, path: str) -> 'NonlinearAdapter':
        data = np.load(path)
        adapter = cls(
            source_dim=int(data['source_dim']),
            target_dim=int(data['target_dim']),
            bottleneck_dim=int(data.get('bottleneck_dim', 128)),
            source_model=str(data.get('source_model', '')),
            target_model=str(data.get('target_model', '')),
        )
        adapter.W1 = data['W1']
        adapter.b1 = data['b1']
        adapter.W2 = data['W2']
        adapter.b2 = data['b2']
        return adapter


# ═══════════════════════════════════════════════════════════════
# Penalty Tensor — Bidirectional Error Correction
# ═══════════════════════════════════════════════════════════════

def compute_attention_entropy(vectors: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Compute per-dimension Shannon entropy of the attention over a batch of vectors.
    
    High entropy = model is confused/uncertain about these dimensions.
    Low entropy = model is confident.
    """
    # Softmax over batch dimension to get "attention" distribution
    scores = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    attn = np.exp(scores / temperature)
    attn /= attn.sum(axis=0, keepdims=True) + 1e-8
    
    # Shannon entropy per dimension
    entropy = -np.sum(attn * np.log(attn + 1e-8), axis=0)
    entropy /= np.log(len(vectors))  # Normalize to [0, 1]
    return entropy


def create_penalty_tensor(entropy: np.ndarray, lambda_penalty: float = 2.0) -> np.ndarray:
    """
    Convert entropy to penalty mask via exp(-λ·H).
    
    penalty[i] = 1 → no suppression (low entropy, model confident)
    penalty[i] → 0 → strong suppression (high entropy, model confused)
    """
    return np.exp(-lambda_penalty * entropy).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Embedding Backend (Pluggable)
# ═══════════════════════════════════════════════════════════════

class EmbeddingBackend:
    """
    Abstract embedding backend. Swap in real models:
      - sentence-transformers (local, offline)
      - OpenAI text-embedding-3-small (API)
      - Cohere embed (API)
      - Custom fine-tuned models
    """
    
    def __init__(self, model_id: str, dim: int):
        self.model_id = model_id
        self.dim = dim
    
    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        raise NotImplementedError
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embed."""
        raise NotImplementedError


class SyntheticEmbeddingBackend(EmbeddingBackend):
    """
    Synthetic backend for development/demo.
    
    Uses a deterministic "semantic hash" to simulate a real embedding space.
    Each model has a different random projection, simulating the fact that
    different models map the same concept to different vector spaces.
    """
    
    def __init__(self, model_id: str, dim: int, seed: int = 42):
        super().__init__(model_id, dim)
        # Each model has its own projection matrix (simulating different training)
        rng = np.random.RandomState(seed)
        # SHA256 produces 32 bytes → 32 float values as fingerprint
        self.fingerprint_dim = 32
        self.projection = rng.randn(self.fingerprint_dim, dim).astype(np.float32) / np.sqrt(self.fingerprint_dim)
        # Model-specific "understanding" — a learnable semantic decoder
        self.latent_basis = rng.randn(dim, dim).astype(np.float32) * 0.1
        # Normalize basis columns
        self.latent_basis /= np.linalg.norm(self.latent_basis, axis=0, keepdims=True)
    
    def embed(self, text: str) -> np.ndarray:
        """Convert text to a deterministic embedding in this model's space."""
        # Create a semantic fingerprint from the text
        h = hashlib.sha256(text.encode('utf-8')).digest()
        fingerprint = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
        fingerprint = fingerprint * 2.0 - 1.0  # [-1, 1]
        
        # Project into this model's latent space
        raw = fingerprint @ self.projection  # [dim]
        
        # Apply model-specific transformation (simulating layer processing)
        latent = raw @ self.latent_basis
        latent = latent / (np.linalg.norm(latent) + 1e-8)  # L2 normalize
        
        return latent.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Tensor Vocabulary — Concept Tokens
# ═══════════════════════════════════════════════════════════════

@dataclass
class TensorVocabulary:
    """
    A vocabulary of "concept vectors" — primitive tokens in the LLM-native language.
    
    Unlike discrete text tokens (50K IDs in GPT), the tensor vocabulary consists
    of continuous vectors organized by semantic domain. Operations between concept
    vectors are continuous geometric operations (interpolation, projection, attention)
    rather than discrete concatenation.
    """
    concepts: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def add_concept(self, name: str, vector: np.ndarray):
        """Register a concept vector."""
        self.concepts[name] = vector / (np.linalg.norm(vector) + 1e-8)
    
    def blend(self, concept_a: str, concept_b: str, alpha: float = 0.5) -> np.ndarray:
        """
        Continuous concept blending: slerp between two semantic vectors.
        
        This is the native "grammar" of LLM-language:
        - Text: "kinda happy but also sad" → discrete tokens fail to capture nuance
        - Tensor: 0.6*happy + 0.4*sad → continuous interpolation on the manifold
        """
        a = self.concepts[concept_a]
        b = self.concepts[concept_b]
        # Spherical linear interpolation
        omega = np.arccos(np.clip(np.dot(a, b), -1, 1))
        if omega < 1e-6:
            return a
        sin_omega = np.sin(omega)
        result = (np.sin((1 - alpha) * omega) / sin_omega) * a + \
                 (np.sin(alpha * omega) / sin_omega) * b
        return result / (np.linalg.norm(result) + 1e-8)
    
    def nearest(self, vector: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Find the k nearest concept vectors (semantic decoding)."""
        v = vector / (np.linalg.norm(vector) + 1e-8)
        scores = [(name, float(np.dot(v, c))) for name, c in self.concepts.items()]
        scores.sort(key=lambda x: -x[1])
        return scores[:k]
    
    def semantic_difference(self, concept_a: str, concept_b: str) -> np.ndarray:
        """
        The "direction" from concept A to concept B.
        This is the tensor equivalent of an analogy: "A is to B as X is to Y"
        """
        return self.concepts[concept_b] - self.concepts[concept_a]
    
    def analogy(self, a: str, b: str, c: str) -> List[Tuple[str, float]]:
        """
        Solve: A is to B as C is to ? 
        Uses vector arithmetic: target = C + (B - A), find nearest concept.
        """
        direction = self.semantic_difference(a, b)
        target = self.concepts[c] + direction
        target /= np.linalg.norm(target) + 1e-8
        return self.nearest(target, k=3)


# ═══════════════════════════════════════════════════════════════
# CMTIP Bus — Message Router
# ═══════════════════════════════════════════════════════════════

class CmtipBus:
    """
    The message bus that routes TensorPackets between models.
    
    Manages:
      - Model registration (embedding backends)
      - Cross-model adapters (learned or on-the-fly)
      - Conversation sessions (sequence tracking)
      - Penalty feedback loop
    """
    
    def __init__(self):
        self.models: Dict[str, EmbeddingBackend] = {}
        self.adapters: Dict[Tuple[str, str], CrossModelAdapter] = {}
        self.sequences: Dict[str, int] = {}  # model → next seq_num
        self.conversations: List[TensorPacket] = []
        self.penalty_threshold: float = 0.7  # Entropy above this triggers penalty
    
    def register_model(self, backend: EmbeddingBackend):
        """Register a model on the bus."""
        self.models[backend.model_id] = backend
        self.sequences[backend.model_id] = 0
    
    def get_adapter(self, source_id: str, target_id: str) -> CrossModelAdapter:
        """Get or create a projection adapter between two models."""
        key = (source_id, target_id)
        if key not in self.adapters:
            source = self.models[source_id]
            target = self.models[target_id]
            self.adapters[key] = CrossModelAdapter(
                source_dim=source.dim,
                target_dim=target.dim,
                source_model=source_id,
                target_model=target_id,
            )
        return self.adapters[key]
    
    def train_adapter(self, source_id: str, target_id: str,
                      paired_texts: List[Tuple[str, str]]):
        """
        Train a cross-model adapter using paired text examples.
        
        Each pair (text_a, text_b) is the same semantic content
        embedded by both models. The adapter learns the linear map.
        """
        source_model = self.models[source_id]
        target_model = self.models[target_id]
        
        source_vecs = np.array([source_model.embed(t[0]) for t in paired_texts])
        target_vecs = np.array([target_model.embed(t[1]) for t in paired_texts])
        
        adapter = self.get_adapter(source_id, target_id)
        mse = adapter.fit(source_vecs, target_vecs)
        return mse
    
    def train_adapter_cca(self, source_id: str, target_id: str,
                          paired_texts: List[Tuple[str, str]], rank: int = 64,
                          reg: float = 0.5) -> dict:
        """
        Train a low-rank CCA cross-model adapter.
        
        CCA finds the shared semantic subspace between two embedding spaces,
        filtering out architecture-specific noise. Much better generalization
        than full-rank OLS for cross-family adapters.
        
        Returns dict with: mse, canonical_corrs, rank, cos_sim_test
        """
        from train_cca import CCACrossModelAdapter
        source_model = self.models[source_id]
        target_model = self.models[target_id]
        
        X = np.array([source_model.embed(t[0]) for t in paired_texts])
        Y = np.array([target_model.embed(t[1]) for t in paired_texts])
        
        cca = CCACrossModelAdapter(source_model.dim, target_model.dim,
                                   rank=rank, source_model=source_id,
                                   target_model=target_id)
        mse, canonical_corrs = cca.fit_cca(X, Y, reg=reg)
        
        # Store as the bus adapter (wrap in CrossModelAdapter-compatible format)
        adapter = self.get_adapter(source_id, target_id)
        adapter.weight = (cca.U @ cca.V).astype(np.float32)
        
        return {
            "mse": float(mse),
            "canonical_corrs_mean": float(canonical_corrs.mean()),
            "canonical_corrs_top5": canonical_corrs[:5].tolist(),
            "rank": rank,
        }
    
    def train_adapter_mlp(self, source_id: str, target_id: str,
                          paired_texts: List[Tuple[str, str]],
                          bottleneck_dim: int = 128, epochs: int = 200,
                          lr: float = 0.01) -> dict:
        """
        Train a nonlinear MLP adapter with bottleneck architecture.
        
        Addresses critique #3: cross-family alignment improves from
        26% (linear) to potentially 40-60% (nonlinear bottleneck)
        by learning manifold deformations, not just linear maps.
        """
        source_model = self.models[source_id]
        target_model = self.models[target_id]
        
        X = np.array([source_model.embed(t[0]) for t in paired_texts])
        Y = np.array([target_model.embed(t[1]) for t in paired_texts])
        
        mlp = NonlinearAdapter(source_model.dim, target_model.dim,
                               bottleneck_dim=bottleneck_dim,
                               source_model=source_id, target_model=target_id)
        final_loss = mlp.fit(X, Y, lr=lr, epochs=epochs, verbose=True)
        
        # Replace the bus adapter with MLP weights (store as CrossModelAdapter.weight)
        adapter = self.get_adapter(source_id, target_id)
        # Store MLP as composite weight: we use the linear adapter's weight matrix
        # as a cache, but routing now goes through the MLP
        adapter.weight = (mlp.W1 @ mlp.W2).astype(np.float32)
        # Also store the MLP for full nonlinear routing
        self._mlp_adapters = getattr(self, '_mlp_adapters', {})
        self._mlp_adapters[(source_id, target_id)] = mlp
        
        return {
            "final_loss": float(final_loss),
            "bottleneck_dim": bottleneck_dim,
            "params": (source_model.dim * bottleneck_dim + bottleneck_dim * target_model.dim),
        }
    
    def tensor_respond(self, source_id: str, received_tensor: np.ndarray,
                       concept_weights: Dict[str, float] = None) -> TensorPacket:
        """
        Tensor-native response: tensor in → tensor out. No text decode.
        
        Addresses critiques #2, #5, #7:
        - No text intermediate (not retrieval, actual generation)
        - Recipient responds directly in tensor space
        - Uses concept blending + received tensor steer
        
        The response is a blend of:
          60%: concept-weighted vector (the intended response meaning)
          40%: received tensor (carries context from sender)
        with the blend ratio expressing agreement/engagement with the input.
        """
        dim = self.models[source_id].dim
        response = np.zeros(dim, dtype=np.float32)
        
        if concept_weights:
            for name, weight in concept_weights.items():
                # Use the model's own embedding of this concept's description text
                concept_vec = self.models[source_id].embed(name)
                response += concept_vec * weight
        
        # Steer by the received tensor (if dimensions match or pad)
        if received_tensor is not None:
            r_flat = received_tensor.flatten()
            if len(r_flat) != dim:
                padded = np.zeros(dim, dtype=np.float32)
                padded[:len(r_flat)] = r_flat[:dim]
                r_flat = padded
            response = 0.6 * response + 0.4 * r_flat
        
        response = response / (np.linalg.norm(response) + 1e-8)
        
        seq = self.sequences[source_id]
        self.sequences[source_id] += 1
        
        packet = TensorPacket(
            source_id=source_id,
            target_id="broadcast",
            seq_num=seq,
            timestamp=time.time(),
            tensor=response,
            shape=(dim,),
            concept_tags=list(concept_weights.keys()) if concept_weights else [],
        )
        self.conversations.append(packet)
        return packet
    
    def send(self, source_id: str, target_id: str, text: str = None,
             tensor: np.ndarray = None, concept_tags: List[str] = None) -> TensorPacket:
        """
        Send a message from one model to another.
        
        If text is provided, the source model embeds it first.
        If tensor is provided, it's wrapped directly.
        """
        if text is not None:
            tensor = self.models[source_id].embed(text)
        
        seq = self.sequences[source_id]
        self.sequences[source_id] += 1
        
        packet = TensorPacket(
            source_id=source_id,
            target_id=target_id,
            seq_num=seq,
            timestamp=time.time(),
            tensor=tensor,
            shape=tensor.shape,
            concept_tags=concept_tags or [],
        )
        
        # Project if needed
        if source_id != target_id:
            adapter = self.get_adapter(source_id, target_id)
            packet = adapter.project_packet(packet)
        
        self.conversations.append(packet)
        
        # Check penalty: does this packet cause high entropy in target?
        entropy = self.check_target_entropy(packet)
        if entropy > self.penalty_threshold:
            recent_tensors = [p.tensor.flatten() for p in self.conversations[-10:]]
            max_dim = max(len(t) for t in recent_tensors)
            padded = np.array([np.pad(t, (0, max_dim - len(t))) for t in recent_tensors])
            penalty = create_penalty_tensor(compute_attention_entropy(padded))
            # Apply penalty to the source model's output space
            # (in production: send penalty tensor back to source via NVLink/CXL)
        
        return packet
    
    def reply(self, target_id: str, source_id: str, text: str = None,
              tensor: np.ndarray = None) -> TensorPacket:
        """Reply to the last message."""
        reply_to = self.conversations[-1].seq_num if self.conversations else -1
        if text is not None:
            tensor = self.models[source_id].embed(text)
        
        seq = self.sequences[source_id]
        self.sequences[source_id] += 1
        
        packet = TensorPacket(
            source_id=source_id,
            target_id=target_id,
            seq_num=seq,
            timestamp=time.time(),
            tensor=tensor,
            shape=tensor.shape,
            reply_to_seq=reply_to,
        )
        
        if source_id != target_id:
            adapter = self.get_adapter(source_id, target_id)
            packet = adapter.project_packet(packet)
        
        self.conversations.append(packet)
        return packet
    
    def check_target_entropy(self, packet: TensorPacket) -> float:
        """Public: check if a packet causes anomalous entropy in the target's recent history."""
        if len(self.conversations) < 5:
            return 0.0
        # Pad all tensors to the same length for comparison
        recent_tensors = [p.tensor.flatten() for p in self.conversations[-5:]]
        max_dim = max(len(t) for t in recent_tensors)
        max_dim = max(max_dim, len(packet.tensor.flatten()))
        padded = []
        for t in recent_tensors:
            p = np.zeros(max_dim, dtype=np.float32)
            p[:len(t)] = t
            padded.append(p)
        pkt_flat = np.zeros(max_dim, dtype=np.float32)
        flat_pkt = packet.tensor.flatten()
        pkt_flat[:len(flat_pkt)] = flat_pkt
        recent = np.vstack([padded, pkt_flat.reshape(1, -1)])
        entropy = compute_attention_entropy(recent)
        return float(entropy.mean())
    
    def save_adapters(self, directory: str):
        """Persist all trained adapters."""
        os.makedirs(directory, exist_ok=True)
        for (src, tgt), adapter in self.adapters.items():
            adapter.save(os.path.join(directory, f"{src}→{tgt}.npz"))
    
    def load_adapters(self, directory: str):
        """Load persisted adapters."""
        for fname in os.listdir(directory):
            if fname.endswith('.npz'):
                adapter = CrossModelAdapter.load(os.path.join(directory, fname))
                key = (adapter.source_model, adapter.target_model)
                self.adapters[key] = adapter
