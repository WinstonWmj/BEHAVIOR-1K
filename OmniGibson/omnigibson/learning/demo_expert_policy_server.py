import argparse
import asyncio
import functools
import http
import importlib.util
import logging
import msgpack
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import websockets

try:
    import websockets.asyncio.server as ws_async_server
except Exception:
    import websockets.legacy.server as ws_async_server


logger = logging.getLogger("demo_expert_policy_server")
logger.setLevel(logging.INFO)


def _load_eval_utils_helpers():
    """
    Load eval_utils directly from its source file so this lightweight websocket
    server can reuse the shared subtask helpers without importing the full
    `omnigibson` package initialization chain.
    """
    module_path = Path(__file__).resolve().parent / "utils" / "eval_utils.py"
    spec = importlib.util.spec_from_file_location("demo_expert_eval_utils", module_path)
    assert spec is not None and spec.loader is not None, f"Failed to load eval_utils from {module_path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.TASK_NAMES_TO_INDICES,
        module.resolve_subtask_frame_range,
        module.resolve_subtask_index_range,
    )


TASK_NAMES_TO_INDICES, resolve_subtask_frame_range, resolve_subtask_index_range = _load_eval_utils_helpers()


def pack_array(obj: Any):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj: Any):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


class ParquetDemoReplayPolicy:
    def __init__(
        self,
        parquet_path: Path,
        start_frame: int = 0,
        end_frame: int | None = None,
        max_steps: int | None = None,
        subtask_index: int | None = None,
        subtask_end_index: int | None = None,
    ):
        self.parquet_path = Path(parquet_path)
        assert self.parquet_path.exists(), f"Parquet file not found: {self.parquet_path}"

        df = pd.read_parquet(self.parquet_path)
        assert "action" in df.columns, f"Expected 'action' column in {self.parquet_path}"
        assert 0 <= start_frame < len(df), (
            f"start_frame={start_frame} must be in [0, {len(df) - 1}] for {self.parquet_path}"
        )
        if end_frame is None:
            end_frame = len(df)
        if max_steps is not None:
            assert max_steps > 0, f"max_steps must be positive, got {max_steps}"
            end_frame = min(start_frame + int(max_steps), len(df))
        assert start_frame < end_frame <= len(df), (
            f"Expected start_frame < end_frame <= {len(df)} for {self.parquet_path}, "
            f"got start_frame={start_frame}, end_frame={end_frame}"
        )

        self._actions = [np.asarray(action, dtype=np.float32) for action in df["action"].tolist()]
        self._episode_index = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else None
        self._task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else None
        self._subtask_index = subtask_index
        self._subtask_end_index = subtask_index if subtask_end_index is None else subtask_end_index
        self._max_steps = max_steps
        self._start_frame = int(start_frame)
        self._end_frame = int(end_frame)
        self._cursor = int(start_frame)
        self._last_action = self._actions[self._cursor].copy()
        self._done = False
        self._done_logged = False

        logger.info(
            "Loaded demo expert parquet: path=%s frames=%d episode_index=%s task_index=%s subtask_range=%s-%s start_frame=%d end_frame=%d max_steps=%s",
            self.parquet_path,
            len(self._actions),
            self._episode_index,
            self._task_index,
            self._subtask_index,
            self._subtask_end_index,
            self._start_frame,
            self._end_frame,
            self._max_steps,
        )

    @property
    def metadata(self) -> dict:
        return {
            "policy_name": "demo_expert",
            "parquet_path": str(self.parquet_path),
            "episode_index": self._episode_index,
            "task_index": self._task_index,
            "subtask_index": self._subtask_index,
            "subtask_end_index": self._subtask_end_index,
            "max_steps": self._max_steps,
            "instance_id": None if self._episode_index is None else int((self._episode_index // 10) % 1000),
            "start_frame": self._start_frame,
            "end_frame": self._end_frame,
            "num_actions": len(self._actions),
            "num_selected_actions": self._end_frame - self._start_frame,
        }

    def reset(self) -> None:
        self._cursor = self._start_frame
        self._last_action = self._actions[self._cursor].copy()
        self._done = False
        self._done_logged = False
        logger.info("Demo expert reset to frame %d", self._cursor)

    @property
    def is_done(self) -> bool:
        return self._done

    def act(self, obs: dict) -> np.ndarray:
        del obs
        if self._cursor >= self._end_frame:
            self._done = True
            if not self._done_logged:
                if self._max_steps is not None:
                    logger.info(
                        "Demo expert reached configured max_steps=%d at frame %d (exclusive end_frame=%d).",
                        self._max_steps,
                        self._cursor,
                        self._end_frame,
                    )
                else:
                    logger.info(
                        "Demo expert reached configured subtask end at frame %d (exclusive end_frame=%d).",
                        self._cursor,
                        self._end_frame,
                    )
                self._done_logged = True
            return self._last_action.copy()

        if self._cursor >= len(self._actions):
            logger.warning(
                "Demo expert ran past the end of the parquet (%d actions). Repeating the last action.",
                len(self._actions),
            )
            self._done = True
            return self._last_action.copy()

        action = self._actions[self._cursor].copy()
        self._last_action = action
        self._cursor += 1
        self._done = self._cursor >= self._end_frame
        return action


class DemoExpertWebsocketServer:
    def __init__(self, policy: ParquetDemoReplayPolicy, host: str, port: int):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = policy.metadata

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        logger.info("Starting demo expert websocket server on %s:%d", self._host, self._port)
        async with ws_async_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                result = unpackb(await websocket.recv(), strict_map_key=False)
                if "reset" in result:
                    self._policy.reset()
                    continue
                action = self._policy.act(result)
                await websocket.send(
                    packer.pack(
                        {
                            "action": action,
                            "done": self._policy.is_done,
                            "server_timing": {"infer_ms": 0.0},
                        }
                    )
                )
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break


def _health_check(connection, request):
    if hasattr(request, "path") and request.path == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
    return None


def _resolve_parquet_path(data_dir: Path, task_name: str | None, task_index: int | None, episode_index: int) -> Path:
    if task_index is None:
        assert task_name is not None, "Either task_name or task_index must be provided."
        task_index = TASK_NAMES_TO_INDICES[task_name]

    inferred_task_index = episode_index // 10000
    assert inferred_task_index == task_index, (
        f"Episode {episode_index} belongs to task-{inferred_task_index:04d}, not task-{task_index:04d}."
    )

    return data_dir / "data" / f"task-{task_index:04d}" / f"episode_{episode_index:08d}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve parquet expert actions over the websocket policy API.")
    parser.add_argument("--demo-data-dir", required=True, help="Path to the 2025-challenge-demos root directory.")
    parser.add_argument("--episode-index", required=True, type=int, help="Full episode index, e.g. 10 or 130010.")
    parser.add_argument("--task-name", default=None, help="Optional task name used to sanity-check the episode index.")
    parser.add_argument("--task-index", default=None, type=int, help="Optional task index used to sanity-check the episode index.")
    parser.add_argument("--subtask-index", default=None, type=int, help="Optional starting subtask index used to auto-resolve the parquet frame range.")
    parser.add_argument("--subtask-end-index", default=None, type=int, help="Optional inclusive end index for sequential replay from subtask-index to subtask-end-index.")
    parser.add_argument("--start-frame", default=0, type=int, help="Manual first parquet frame when --subtask-index is omitted.")
    parser.add_argument("--max-steps", default=None, type=int, help="Optional replay step cap. When set, it overrides any resolved subtask end frame.")
    parser.add_argument("--host", default="0.0.0.0", help="Websocket host to bind.")
    parser.add_argument("--port", default=8007, type=int, help="Websocket port to bind.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    resolved_task_index = args.task_index
    if resolved_task_index is None and args.task_name is not None:
        resolved_task_index = TASK_NAMES_TO_INDICES[args.task_name]

    resolved_subtask_range = resolve_subtask_index_range(
        subtask_index=args.subtask_index,
        subtask_end_index=args.subtask_end_index,
    )
    resolved_start_frame = args.start_frame
    resolved_end_frame = None
    if resolved_subtask_range is not None:
        resolved_subtask_start_idx, resolved_subtask_end_idx = resolved_subtask_range
        assert resolved_task_index is not None, "Either --task-name or --task-index is required when using --subtask-index."
        if args.start_frame != 0:
            logger.info(
                "Ignoring explicit start_frame=%d because subtask range %d-%d was provided.",
                args.start_frame,
                resolved_subtask_start_idx,
                resolved_subtask_end_idx,
            )
        resolved_start_frame, resolved_end_frame = resolve_subtask_frame_range(
            demo_data_dir=args.demo_data_dir,
            task_index=resolved_task_index,
            episode_index=args.episode_index,
            subtask_index=resolved_subtask_start_idx,
            subtask_end_index=resolved_subtask_end_idx,
        )
        logger.info(
            "Resolved frame range [%d, %d) from subtask range %d-%d for episode=%d",
            resolved_start_frame,
            resolved_end_frame,
            resolved_subtask_start_idx,
            resolved_subtask_end_idx,
            args.episode_index,
        )
    if args.max_steps is not None and resolved_end_frame is not None:
        logger.info(
            "Ignoring resolved end_frame=%d because max_steps=%d was provided.",
            resolved_end_frame,
            args.max_steps,
        )
        resolved_end_frame = None

    parquet_path = _resolve_parquet_path(
        data_dir=Path(args.demo_data_dir),
        task_name=args.task_name,
        task_index=resolved_task_index,
        episode_index=args.episode_index,
    )
    policy = ParquetDemoReplayPolicy(
        parquet_path=parquet_path,
        start_frame=resolved_start_frame,
        end_frame=resolved_end_frame,
        max_steps=args.max_steps,
        subtask_index=args.subtask_index,
        subtask_end_index=args.subtask_end_index,
    )
    server = DemoExpertWebsocketServer(policy=policy, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
