"""
CMTIP Bus gRPC Client
══════════════════════

Client library + CLI for the CMTIP Bus gRPC service.

Usage:
    python cmtip_client.py [--host localhost:50051] <command> [args...]

Commands:
    status                    Show bus status
    register <id> <dim> [backend_type] [config_json]
                              Register a model on the bus
    send <src> <tgt> <text>   Send text → auto-embed → cross-project
    send-tensor <src> <tgt> <shape_dims> <data_file.npy>
                              Send raw tensor
    train <src> <tgt> <pairs_file.json>
                              Train W_{src→tgt} adapter from paired data
    conversation [limit]      Show recent packets
    subscribe                 Live bidirectional streaming
    benchmark <iterations>    Throughput benchmark
"""

import grpc
import numpy as np
import json
import time
import sys
import argparse

import cmtip_service_pb2 as pb2
import cmtip_service_pb2_grpc as pb2_grpc


class CmtipClient:
    """Python client for the CMTIP Bus gRPC service."""

    def __init__(self, host: str = "localhost:50051"):
        self.channel = grpc.insecure_channel(
            host,
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ],
        )
        self.stub = pb2_grpc.CmtipBusStub(self.channel)

    def status(self) -> dict:
        """Get bus status."""
        resp = self.stub.GetStatus(pb2.StatusRequest())
        return {
            "registered_models": resp.registered_models,
            "trained_adapters": resp.trained_adapters,
            "total_conversations": resp.total_conversations,
            "model_ids": list(resp.model_ids),
            "penalty_threshold": resp.penalty_threshold,
        }

    def register_model(self, model_id: str, dim: int,
                       backend_type: str = "synthetic",
                       backend_config: dict = None) -> dict:
        """Register a model on the bus."""
        resp = self.stub.RegisterModel(pb2.RegisterModelRequest(
            model_id=model_id,
            dim=dim,
            backend_type=backend_type,
            backend_config=json.dumps(backend_config) if backend_config else "",
        ))
        return {"success": resp.success, "error": resp.error, "dim": resp.registered_dim}

    def send_text(self, source_id: str, target_id: str, text: str,
                  concept_tags: list = None) -> dict:
        """Send text through the bus."""
        resp = self.stub.SendText(pb2.SendTextRequest(
            source_id=source_id,
            target_id=target_id,
            text=text,
            concept_tags=concept_tags or [],
        ))
        return {
            "source_id": resp.packet.source_id,
            "target_id": resp.packet.target_id,
            "seq_num": resp.packet.seq_num,
            "shape": list(resp.packet.shape),
            "concept_tags": list(resp.packet.concept_tags),
            "confidence": resp.packet.confidence,
            "penalty_triggered": resp.penalty_triggered,
            "target_entropy": resp.target_entropy,
        }

    def send_tensor(self, source_id: str, target_id: str,
                    tensor: np.ndarray, concept_tags: list = None,
                    confidence: float = 1.0) -> dict:
        """Send raw tensor."""
        resp = self.stub.SendTensor(pb2.SendTensorRequest(
            source_id=source_id,
            target_id=target_id,
            tensor=tensor.astype(np.float32).tobytes(),
            shape=list(tensor.shape),
            concept_tags=concept_tags or [],
            confidence=confidence,
        ))
        return {
            "source_id": resp.packet.source_id,
            "target_id": resp.packet.target_id,
            "seq_num": resp.packet.seq_num,
            "penalty_triggered": resp.penalty_triggered,
            "target_entropy": resp.target_entropy,
        }

    def train_adapter(self, source_id: str, target_id: str,
                      paired_data: list) -> dict:
        """Train a cross-model adapter."""
        resp = self.stub.TrainAdapter(pb2.TrainAdapterRequest(
            source_id=source_id,
            target_id=target_id,
            source_texts=[p[0] for p in paired_data],
            target_texts=[p[1] for p in paired_data],
        ))
        return {
            "mse": resp.mse,
            "num_pairs": resp.num_pairs,
            "saved": resp.saved,
            "adapter_path": resp.adapter_path,
        }

    def get_conversation(self, limit: int = 20) -> list:
        """Get conversation history."""
        resp = self.stub.GetConversation(pb2.GetConversationRequest(limit=limit))
        return [
            {
                "source_id": p.source_id,
                "target_id": p.target_id,
                "seq_num": p.seq_num,
                "shape": list(p.shape),
                "concept_tags": list(p.concept_tags),
                "confidence": p.confidence,
            }
            for p in resp.packets
        ]

    def subscribe(self, on_packet=None):
        """Subscribe to live packet stream (bidirectional)."""
        def packet_generator():
            # Send one empty packet to open the stream, then yield whatever
            yield pb2.TensorPacket()
            # Stay alive to receive
            import time
            while True:
                time.sleep(30)

        stream = self.stub.Subscribe(packet_generator())
        for proto in stream:
            if on_packet:
                on_packet(proto)
            else:
                sys.stdout.write(
                    f"\r[{proto.source_id}→{proto.target_id}] "
                    f"seq={proto.seq_num} tags={list(proto.concept_tags)} "
                    f"shape={list(proto.shape)}              "
                )
                sys.stdout.flush()

    def close(self):
        self.channel.close()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CMTIP Bus gRPC Client")
    parser.add_argument("--host", default="localhost:50051", help="Server host:port")
    sub = parser.add_subparsers(dest="command")

    p_status = sub.add_parser("status", help="Bus status")

    p_register = sub.add_parser("register", help="Register model")
    p_register.add_argument("model_id")
    p_register.add_argument("dim", type=int)
    p_register.add_argument("--backend", default="synthetic")
    p_register.add_argument("--config", default="{}")

    p_send = sub.add_parser("send", help="Send text")
    p_send.add_argument("source_id")
    p_send.add_argument("target_id")
    p_send.add_argument("text")
    p_send.add_argument("--tags", nargs="*", default=[])

    p_send_t = sub.add_parser("send-tensor", help="Send raw tensor")
    p_send_t.add_argument("source_id")
    p_send_t.add_argument("target_id")
    p_send_t.add_argument("shape_dims", help="Comma-separated dims, e.g. 384")
    p_send_t.add_argument("--file", help=".npy file with tensor data")
    p_send_t.add_argument("--tags", nargs="*", default=[])

    p_train = sub.add_parser("train", help="Train adapter")
    p_train.add_argument("source_id")
    p_train.add_argument("target_id")
    p_train.add_argument("pairs_file", help="JSON file: [[src_text, tgt_text], ...]")

    p_conv = sub.add_parser("conversation", help="Show conversation")
    p_conv.add_argument("limit", type=int, nargs="?", default=10)

    p_bench = sub.add_parser("benchmark", help="Throughput benchmark")
    p_bench.add_argument("iterations", type=int, nargs="?", default=100)

    args = parser.parse_args()
    client = CmtipClient(args.host)

    try:
        if args.command == "status":
            s = client.status()
            print(f"Registered models:  {s['registered_models']}")
            print(f"Trained adapters:   {s['trained_adapters']}")
            print(f"Total conversations:{s['total_conversations']}")
            print(f"Model IDs:          {s['model_ids']}")
            print(f"Penalty threshold:  {s['penalty_threshold']}")

        elif args.command == "register":
            config = json.loads(args.config)
            r = client.register_model(args.model_id, args.dim, args.backend, config)
            if r["success"]:
                print(f"Registered {args.model_id} (dim={r['dim']})")
            else:
                print(f"ERROR: {r['error']}")
                sys.exit(1)

        elif args.command == "send":
            r = client.send_text(args.source_id, args.target_id, args.text, args.tags)
            print(f"Sent: {r['source_id']}→{r['target_id']} seq={r['seq_num']}")
            print(f"  Shape: {r['shape']}  Tags: {r['concept_tags']}")
            print(f"  Confidence: {r['confidence']:.3f}")
            print(f"  Penalty triggered: {r['penalty_triggered']} (entropy={r['target_entropy']:.3f})")

        elif args.command == "send_tensor":
            dims = [int(d) for d in args.shape_dims.split(",")]
            if args.file:
                tensor = np.load(args.file)
            else:
                tensor = np.random.randn(*dims).astype(np.float32)
            r = client.send_tensor(args.source_id, args.target_id, tensor, args.tags)
            print(f"Sent tensor: {r['source_id']}→{r['target_id']} seq={r['seq_num']}")
            print(f"  Penalty: {r['penalty_triggered']} (entropy={r['target_entropy']:.3f})")

        elif args.command == "train":
            with open(args.pairs_file) as f:
                pairs = json.load(f)
            r = client.train_adapter(args.source_id, args.target_id, pairs)
            print(f"Trained {args.source_id}→{args.target_id}")
            print(f"  Pairs: {r['num_pairs']}  MSE: {r['mse']:.6f}")
            print(f"  Saved: {r['saved']} ({r['adapter_path']})")

        elif args.command == "conversation":
            conv = client.get_conversation(args.limit)
            print(f"Conversation ({len(conv)} packets):")
            for p in conv:
                print(f"  #{p['seq_num']:03d}: {p['source_id']}→{p['target_id']} "
                      f"shape={p['shape']} tags={p['concept_tags']} conf={p['confidence']:.2f}")

        elif args.command == "benchmark":
            # Auto-register models needed for benchmark
            model_count = 4
            for m in range(model_count):
                mid = f"synthetic-{m}"
                client.register_model(mid, 256, "synthetic")

            print(f"Benchmarking {args.iterations} iterations...")
            latency_us = []
            start_total = time.perf_counter()
            for i in range(args.iterations):
                t0 = time.perf_counter()
                client.send_text(f"synthetic-{i%model_count}", f"synthetic-{(i+1)%model_count}",
                                 f"benchmark message {i}")
                latency_us.append((time.perf_counter() - t0) * 1e6)
            total = time.perf_counter() - start_total

            lat = np.array(latency_us)
            print(f"  Total:    {total:.3f}s")
            print(f"  Throughput: {args.iterations/total:.1f} msg/s")
            print(f"  Latency p50: {np.percentile(lat, 50):.0f} µs")
            print(f"  Latency p95: {np.percentile(lat, 95):.0f} µs")
            print(f"  Latency p99: {np.percentile(lat, 99):.0f} µs")

        else:
            parser.print_help()

    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()} — {e.details()}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
