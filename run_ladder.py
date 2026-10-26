"""Compatibility entry point for the existing ladder runner."""

from runLadderGames import main

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
