"""Install and manage a loopback-only Showdown process for local bot games."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

log = logging.getLogger(__name__)
REPOSITORY = "https://github.com/smogon/pokemon-showdown.git"
DEFAULT_DIRECTORY = Path(__file__).resolve().parent / "data" / "pokemon-showdown"
URL = "ws://localhost:8000/showdown/websocket"
MIN_NODE = (22, 18, 0)


async def server_ready() -> bool:
    """A listening TCP socket alone is not proof this is a Showdown server."""
    try:
        async with asyncio.timeout(2):
            async with connect(URL, open_timeout=1, close_timeout=0.2, proxy=None) as websocket:
                async for message in websocket:
                    if isinstance(message, str) and any(
                        line.startswith("|challstr|") for line in message.splitlines()
                    ):
                        return True
    except (OSError, TimeoutError, WebSocketException):
        pass
    return False


async def port_in_use() -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection("localhost", 8000), 1)
    except (OSError, TimeoutError):
        return False
    writer.close()
    await writer.wait_closed()
    return True


async def stop_process(process: asyncio.subprocess.Process) -> None:
    """Stop the owned process group, including Showdown's simulator workers."""
    def send(sig):
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif process.returncode is None:
                process.terminate() if sig == signal.SIGTERM else process.kill()
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), 5)
    except TimeoutError:
        send(signal.SIGKILL)
        await process.wait()


async def run_setup(*command: str, cwd: Path, log_path: Path) -> None:
    log.info("Showdown setup: %s (details: %s)", " ".join(command[:3]), log_path)
    with log_path.open("ab") as stream:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=cwd, stdout=stream, stderr=asyncio.subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        try:
            code = await process.wait()
        except BaseException:
            await stop_process(process)
            raise
    if code:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(f"Showdown setup failed (exit {code}); see {log_path}\n{tail}")


async def setup_server(directory: Path) -> str:
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Local games require Node.js 22.18.0 or newer. Install Node.js, then rerun --local.")
    process = await asyncio.create_subprocess_exec(node, "--version", stdout=asyncio.subprocess.PIPE)
    stdout, _ = await process.communicate()
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", stdout.decode())
    if process.returncode or not match or tuple(map(int, match.groups())) < MIN_NODE:
        raise RuntimeError("Local games require Node.js 22.18.0 or newer; found " + stdout.decode().strip())
    directory.parent.mkdir(parents=True, exist_ok=True)
    setup_log = directory.parent / f"{directory.name}-setup.log"
    if not directory.exists():
        git = shutil.which("git")
        if not git:
            raise RuntimeError("Git is required to download Pokémon Showdown. Install Git, then rerun --local.")
        # Publish only successful clones, so a cancelled first run can be retried.
        with tempfile.TemporaryDirectory(prefix="showdown-download-", dir=directory.parent) as temporary:
            clone = Path(temporary) / "server"
            await run_setup(git, "clone", "--depth", "1", REPOSITORY, str(clone),
                            cwd=directory.parent, log_path=setup_log)
            clone.rename(directory)
    if not (directory / "pokemon-showdown").is_file() or not (directory / "package.json").is_file():
        raise RuntimeError(f"Not a Showdown checkout: {directory}. Choose a checkout with --showdown-dir.")
    marker = directory / "node_modules" / ".vgc-installed"
    if not marker.exists():
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("npm is required to install Showdown dependencies. Install Node.js with npm.")
        await run_setup(npm, "install", "--omit=optional", "--no-audit", "--no-fund",
                        cwd=directory, log_path=setup_log)
        marker.touch()
    config = directory / "config" / "config.js"
    if not config.exists():
        example = directory / "config" / "config-example.js"
        config.write_text(example.read_text() + "\n// Local bot server: bind only to this computer.\n"
                          "exports.bindaddress = '127.0.0.1';\nexports.port = 8000;\n")
    return node


@asynccontextmanager
async def local_showdown(directory: str | Path = DEFAULT_DIRECTORY, startup_timeout: float = 120):
    """Reuse an existing server, or provision/start one and stop it on exit."""
    if await server_ready():
        log.info("Using existing Showdown server on localhost:8000")
        yield
        return
    if await port_in_use():
        raise RuntimeError("Port 8000 is occupied but no Showdown handshake was received. "
                           "Stop the other service and rerun --local.")
    directory = Path(directory).expanduser().resolve()
    node = await setup_server(directory)
    # Another process may have started a server while dependencies installed.
    if await server_ready():
        log.info("Using Showdown server started during setup on localhost:8000")
        yield
        return
    log_path = directory.parent / f"{directory.name}-server.log"
    # Preload also applies to Node workers, without rewriting an existing config.
    preload = directory.parent / f"{directory.name}-loopback.cjs"
    preload.write_text("const config = require(process.cwd() + '/config/config.js');\n"
                       "config.bindaddress = '127.0.0.1';\nconfig.ssl = null;\n")
    log.info("Starting Showdown on localhost:8000 (log: %s)", log_path)
    with log_path.open("ab") as stream:
        process = await asyncio.create_subprocess_exec(
            node, "--require", str(preload), "pokemon-showdown", "start", "8000", "--no-security",
            cwd=directory, stdout=stream, stderr=asyncio.subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        try:
            deadline = asyncio.get_running_loop().time() + startup_timeout
            while process.returncode is None:
                if await server_ready():
                    log.info("Showdown ready; starting bot games")
                    yield
                    return
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(f"Showdown did not become ready within {startup_timeout:g}s; see {log_path}")
                await asyncio.sleep(0.25)
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
            raise RuntimeError(f"Showdown exited with code {process.returncode}; see {log_path}\n{tail}")
        finally:
            await stop_process(process)
            log.info("Stopped the Showdown server started for this run")
