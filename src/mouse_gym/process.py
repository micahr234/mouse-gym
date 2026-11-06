"""Spawned environment owners with parent-owned, reusable shared-memory slots.

Pipes carry commands, sizes, and errors. Config factories and both action/result
payloads use shared memory. Pickle protocol 5 keeps NumPy buffers out of the
metadata stream; readers copy them before the slot can be reused.
"""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection, wait
from multiprocessing.shared_memory import SharedMemory
import pickle
import traceback
from typing import Any, cast

import cloudpickle


def _pack(value: Any) -> list[bytes | memoryview]:
    buffers: list[pickle.PickleBuffer] = []
    metadata = cloudpickle.dumps(value, protocol=5, buffer_callback=buffers.append)
    return [metadata, *(buffer.raw() for buffer in buffers)]


def _write(memory: SharedMemory, parts: list[bytes | memoryview]) -> list[int]:
    assert memory.buf is not None
    offset = 0
    for part in parts:
        memory.buf[offset:offset + len(part)] = part
        offset += len(part)
    return [len(part) for part in parts]


def _read(memory: SharedMemory, sizes: list[int]) -> Any:
    assert memory.buf is not None
    offset = sizes[0]
    buffers = []
    for size in sizes[1:]:
        # Detach before either side reuses the slot, including arrays retained
        # by an environment or by the caller across subsequent step calls.
        buffers.append(bytearray(memory.buf[offset:offset + size]))
        offset += size
    return pickle.loads(bytes(memory.buf[:sizes[0]]), buffers=buffers)


def _worker(connection: Connection, name: str, sizes: list[int]) -> None:
    from mouse_gym.build import make_env

    envs = {}
    memory = None
    closed = set()
    try:
        memory = SharedMemory(name=name, track=False)
        configs = _read(memory, sizes)
        for index, config in configs:
            envs[index] = make_env(config)
        result = [(index, (env.name, env.action_space, env.observation_space))
                  for index, env in envs.items()]
        while True:
            parts = _pack(result)
            required = sum(map(len, parts))
            if required > memory.size:
                connection.send(("resize", required))
                memory.close()
                memory = SharedMemory(name=connection.recv(), track=False)
            connection.send(("ready", _write(memory, parts)))
            del parts, result
            job = connection.recv()
            if job is None:
                break
            command, name, sizes = job
            if name != memory.name:
                memory.close()
                memory = SharedMemory(name=name, track=False)
            requests = _read(memory, sizes)
            result = []
            for index, argument in requests:
                if index in closed:
                    raise RuntimeError(f"Environment {index} is closed")
                if command == "step":
                    output = envs[index].step(argument)
                elif command == "render":
                    output = envs[index].render()
                elif command == "close_env":
                    envs[index].close()
                    closed.add(index)
                    output = None
                else:
                    raise ValueError(f"Unknown environment command {command!r}")
                result.append((index, output))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        for index, env in envs.items():
            if index not in closed:
                try:
                    env.close()
                except Exception:
                    pass
        if memory is not None:
            memory.close()
        connection.close()


class ProcessEnvs:
    """Each process owns a fixed shard of envs for the group's lifetime."""

    def __init__(self, configs, num_workers: int):
        self._processes = []
        self._connections: list[Connection] = []
        self._slots: list[SharedMemory] = []
        self._closed = False
        self._count = min(num_workers, len(configs))
        self._owners = {i: i % self._count for i in range(len(configs))}
        context = mp.get_context("spawn")
        try:
            for worker in range(self._count):
                parts = _pack([(i, config) for i, config in enumerate(configs)
                               if self._owners[i] == worker])
                memory = SharedMemory(create=True, size=max(65536, sum(map(len, parts))))
                self._slots.append(memory)
                parent, child = context.Pipe()
                self._connections.append(parent)
                process = context.Process(target=_worker, args=(child, memory.name, _write(memory, parts)))
                try:
                    process.start()
                finally:
                    child.close()
                self._processes.append(process)
            self.metadata = self._receive(list(range(self._count)))
        except BaseException:
            self.close()
            raise

    def _resize(self, worker: int, required: int) -> SharedMemory:
        old = self._slots[worker]
        if required > old.size:
            memory = SharedMemory(create=True, size=max(65536, 1 << (required - 1).bit_length()))
            self._slots[worker] = memory
            old.close()
            old.unlink()
        return self._slots[worker]

    def _receive(self, workers: list[int]) -> dict[int, Any]:
        pending = {self._connections[i]: i for i in workers}
        result = {}
        while pending:
            for item in wait(list(pending), timeout=0.05):
                connection = cast(Connection, item)
                worker = pending[connection]
                try:
                    message, payload = connection.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError("Environment worker disconnected unexpectedly") from exc
                if message == "error":
                    raise RuntimeError("Environment worker failed:\n" + payload)
                if message == "resize":
                    connection.send(self._resize(worker, payload).name)
                elif message == "ready":
                    result.update(_read(self._slots[worker], payload))
                    del pending[connection]
                else:
                    raise RuntimeError(f"Unexpected environment worker message {message!r}")
            for worker in pending.values():
                if not self._processes[worker].is_alive():
                    raise RuntimeError(f"Environment worker exited unexpectedly ({self._processes[worker].exitcode})")
        return result

    def call(self, command: str, requests: dict[int, Any]) -> dict[int, Any]:
        if self._closed:
            raise RuntimeError("GroupEnv is closed")
        shards: dict[int, list] = {}
        for index, argument in requests.items():
            shards.setdefault(self._owners[index], []).append((index, argument))
        try:
            for worker, values in shards.items():
                parts = _pack(values)
                memory = self._resize(worker, sum(map(len, parts)))
                self._connections[worker].send((command, memory.name, _write(memory, parts)))
            return self._receive(list(shards))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        for connection in self._connections:
            connection.close()
        for memory in self._slots:
            memory.close()
            memory.unlink()

    def __del__(self):
        if hasattr(self, "_closed"):
            self.close()
