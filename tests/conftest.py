# SPDX-License-Identifier: Apache-2.0

import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

import httpx
import pytest
from vllm.plugins import load_general_plugins


@pytest.fixture(scope="session", autouse=True)
def load_plugin():
    load_general_plugins()


@pytest.fixture
def run_server(tmp_path):
    @contextmanager
    def start(model, args):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        log_path = tmp_path / "server.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vllm.entrypoints.cli.main",
                    "serve",
                    model,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    *args,
                ],
                env={
                    **os.environ,
                    "VLLM_USE_V2_MODEL_RUNNER": "1",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 300
                with httpx.Client(timeout=1, trust_env=False) as client:
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            pytest.fail(log_path.read_text())
                        try:
                            if client.get(url + "/health").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(0.5)
                    else:
                        pytest.fail(
                            "Server startup timed out:\n" + log_path.read_text()
                        )
                yield url + "/v1"
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()

    return start
