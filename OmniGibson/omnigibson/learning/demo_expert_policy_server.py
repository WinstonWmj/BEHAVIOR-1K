import argparse
import asyncio
import functools
import http
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


TASK_NAMES_TO_INDICES = {
    "turning_on_radio": 0,
    "picking_up_trash": 1,
    "putting_away_Halloween_decorations": 2,
    "cleaning_up_plates_and_food": 3,
    "can_meat": 4,
    "setting_mousetraps": 5,
    "hiding_Easter_eggs": 6,
    "picking_up_toys": 7,
    "rearranging_kitchen_furniture": 8,
    "putting_up_Christmas_decorations_inside": 9,
    "set_up_a_coffee_station_in_your_kitchen": 10,
    "putting_dishes_away_after_cleaning": 11,
    "preparing_lunch_box": 12,
    "loading_the_car": 13,
    "carrying_in_groceries": 14,
    "bringing_in_wood": 15,
    "moving_boxes_to_storage": 16,
    "bringing_water": 17,
    "tidying_bedroom": 18,
    "outfit_a_basic_toolbox": 19,
    "sorting_vegetables": 20,
    "collecting_childrens_toys": 21,
    "putting_shoes_on_rack": 22,
    "boxing_books_up_for_storage": 23,
    "storing_food": 24,
    "clearing_food_from_table_into_fridge": 25,
    "assembling_gift_baskets": 26,
    "sorting_household_items": 27,
    "getting_organized_for_work": 28,
    "clean_up_your_desk": 29,
    "setting_the_fire": 30,
    "clean_boxing_gloves": 31,
    "wash_a_baseball_cap": 32,
    "wash_dog_toys": 33,
    "hanging_pictures": 34,
    "attach_a_camera_to_a_tripod": 35,
    "clean_a_patio": 36,
    "clean_a_trumpet": 37,
    "spraying_for_bugs": 38,
    "spraying_fruit_trees": 39,
    "make_microwave_popcorn": 40,
    "cook_cabbage": 41,
    "chop_an_onion": 42,
    "slicing_vegetables": 43,
    "chopping_wood": 44,
    "cook_hot_dogs": 45,
    "cook_bacon": 46,
    "freeze_pies": 47,
    "canning_food": 48,
    "make_pizza": 49,
}


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
    def __init__(self, parquet_path: Path, start_frame: int = 0):
        self.parquet_path = Path(parquet_path)
        assert self.parquet_path.exists(), f"Parquet file not found: {self.parquet_path}"

        df = pd.read_parquet(self.parquet_path)
        assert "action" in df.columns, f"Expected 'action' column in {self.parquet_path}"
        assert 0 <= start_frame < len(df), (
            f"start_frame={start_frame} must be in [0, {len(df) - 1}] for {self.parquet_path}"
        )

        self._actions = [np.asarray(action, dtype=np.float32) for action in df["action"].tolist()]
        self._episode_index = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else None
        self._task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else None
        self._start_frame = int(start_frame)
        self._cursor = int(start_frame)
        self._last_action = self._actions[self._cursor].copy()

        logger.info(
            "Loaded demo expert parquet: path=%s frames=%d episode_index=%s task_index=%s start_frame=%d",
            self.parquet_path,
            len(self._actions),
            self._episode_index,
            self._task_index,
            self._start_frame,
        )

    @property
    def metadata(self) -> dict:
        return {
            "policy_name": "demo_expert",
            "parquet_path": str(self.parquet_path),
            "episode_index": self._episode_index,
            "task_index": self._task_index,
            "instance_id": None if self._episode_index is None else int((self._episode_index // 10) % 1000),
            "start_frame": self._start_frame,
            "num_actions": len(self._actions),
        }

    def reset(self) -> None:
        self._cursor = self._start_frame
        self._last_action = self._actions[self._cursor].copy()
        logger.info("Demo expert reset to frame %d", self._cursor)

    def act(self, obs: dict) -> np.ndarray:
        del obs
        if self._cursor >= len(self._actions):
            logger.warning(
                "Demo expert ran past the end of the parquet (%d actions). Repeating the last action.",
                len(self._actions),
            )
            return self._last_action.copy()

        action = self._actions[self._cursor].copy()
        self._last_action = action
        self._cursor += 1
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
    parser.add_argument("--start-frame", default=0, type=int, help="First parquet frame / action index to replay.")
    parser.add_argument("--host", default="0.0.0.0", help="Websocket host to bind.")
    parser.add_argument("--port", default=8007, type=int, help="Websocket port to bind.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parquet_path = _resolve_parquet_path(
        data_dir=Path(args.demo_data_dir),
        task_name=args.task_name,
        task_index=args.task_index,
        episode_index=args.episode_index,
    )
    policy = ParquetDemoReplayPolicy(parquet_path=parquet_path, start_frame=args.start_frame)
    server = DemoExpertWebsocketServer(policy=policy, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
