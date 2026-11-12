import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import local_server as S


class LocalServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_server_is_reused_and_never_stopped(self):
        with patch.object(S, 'server_ready', return_value=True), \
                patch.object(S, 'setup_server') as setup, patch.object(S, 'stop_process') as stop:
            async with S.local_showdown():
                pass
        setup.assert_not_called()
        stop.assert_not_called()

    async def test_occupied_port_does_not_install_or_start_server(self):
        with patch.object(S, 'server_ready', return_value=False), \
                patch.object(S, 'port_in_use', return_value=True), \
                patch.object(S, 'setup_server') as setup:
            with self.assertRaisesRegex(RuntimeError, 'Port 8000 is occupied'):
                async with S.local_showdown():
                    self.fail('bots must not start')
        setup.assert_not_called()

    async def test_owned_server_stops_when_bots_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'showdown'
            directory.mkdir()
            process = SimpleNamespace(returncode=None)
            with patch.object(S, 'server_ready', side_effect=[False, False, True]), \
                    patch.object(S, 'port_in_use', return_value=False), \
                    patch.object(S, 'setup_server', return_value='node'), \
                    patch.object(S.asyncio, 'create_subprocess_exec', return_value=process) as start, \
                    patch.object(S, 'stop_process') as stop:
                with self.assertRaisesRegex(RuntimeError, 'battle failed'):
                    async with S.local_showdown(directory):
                        raise RuntimeError('battle failed')
            stop.assert_awaited_once_with(process)
            self.assertIn('--no-security', start.call_args.args)
            preload = Path(start.call_args.args[2])
            self.assertIn("127.0.0.1", preload.read_text())

    async def test_startup_timeout_cleans_up_and_does_not_run_bots(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'showdown'
            directory.mkdir()
            process = SimpleNamespace(returncode=None)
            with patch.object(S, 'server_ready', return_value=False), \
                    patch.object(S, 'port_in_use', return_value=False), \
                    patch.object(S, 'setup_server', return_value='node'), \
                    patch.object(S.asyncio, 'create_subprocess_exec', return_value=process), \
                    patch.object(S, 'stop_process') as stop:
                with self.assertRaisesRegex(RuntimeError, 'did not become ready'):
                    async with S.local_showdown(directory, startup_timeout=0):
                        self.fail('bots must not start')
            stop.assert_awaited_once_with(process)

    async def test_setup_installs_once_and_preserves_existing_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'showdown'
            directory.mkdir()
            (directory / 'pokemon-showdown').touch()
            (directory / 'package.json').write_text('{}')
            (directory / 'config').mkdir()
            config = directory / 'config' / 'config.js'
            config.write_text('exports.custom = true;')
            process = SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b'v25.8.1', b'')))

            async def install(*args, **kwargs):
                (directory / 'node_modules').mkdir(exist_ok=True)

            with patch.object(S.shutil, 'which', side_effect=lambda tool: tool), \
                    patch.object(S.asyncio, 'create_subprocess_exec', return_value=process), \
                    patch.object(S, 'run_setup', side_effect=install) as setup:
                await S.setup_server(directory)
                await S.setup_server(directory)
            self.assertEqual(setup.await_count, 1)
            self.assertEqual(setup.call_args.args[:2], ('npm', 'install'))
            self.assertEqual(config.read_text(), 'exports.custom = true;')

    async def test_download_clone_published_only_after_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'showdown'
            process = SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b'v25.8.1', b'')))

            async def command(*args, **kwargs):
                if args[0] == 'git':
                    checkout = Path(args[-1])
                    checkout.mkdir()
                    (checkout / 'pokemon-showdown').touch()
                    (checkout / 'package.json').write_text('{}')
                    (checkout / 'config').mkdir()
                    (checkout / 'config' / 'config-example.js').write_text('exports.port = 8000;')
                else:
                    (kwargs['cwd'] / 'node_modules').mkdir()

            with patch.object(S.shutil, 'which', side_effect=lambda tool: tool), \
                    patch.object(S.asyncio, 'create_subprocess_exec', return_value=process), \
                    patch.object(S, 'run_setup', side_effect=command):
                await S.setup_server(directory)
            self.assertTrue((directory / 'pokemon-showdown').exists())
            self.assertIn('127.0.0.1', (directory / 'config' / 'config.js').read_text())
            self.assertFalse(list(Path(temporary).glob('showdown-download-*')))

    async def test_missing_node_has_actionable_error(self):
        with patch.object(S.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Install Node.js'):
                await S.setup_server(Path('/unused'))

    async def test_old_node_fails_before_install(self):
        process = SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b'v20.0.0', b'')))
        with patch.object(S.shutil, 'which', return_value='node'), \
                patch.object(S.asyncio, 'create_subprocess_exec', return_value=process), \
                patch.object(S, 'run_setup') as setup:
            with self.assertRaisesRegex(RuntimeError, '22.18.0'):
                await S.setup_server(Path('/unused'))
        setup.assert_not_called()


if __name__ == '__main__':
    unittest.main()
