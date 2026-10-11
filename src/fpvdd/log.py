"""Logging setup shared by all scripts."""

import logging


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging; DEBUG if ``verbose`` else INFO."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
