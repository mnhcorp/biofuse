"""
Logging utilities for BioFuse.

Provides structured logging functionality for experiments, training,
and evaluation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = 'biofuse',
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and/or file output.

    Args:
        name: Logger name
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional path to log file
        console: Whether to log to console

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger('biofuse', level=logging.DEBUG)
        >>> logger.info('Starting training...')
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentLogger:
    """
    Logger specifically designed for experiment tracking.

    Logs experiment parameters, metrics, and results in a structured way.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: Optional[Path] = None,
        console: bool = True
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to store log files
            console: Whether to also log to console
        """
        self.experiment_name = experiment_name
        self.start_time = datetime.now()

        # Set up log file
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{experiment_name}_{timestamp}.log"
        else:
            log_file = None

        self.logger = setup_logger(
            f'biofuse.{experiment_name}',
            log_file=log_file,
            console=console
        )

        self.metrics = {}

    def log_params(self, params: dict):
        """
        Log experiment parameters.

        Args:
            params: Dictionary of parameter names and values
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info("=" * 80)
        self.logger.info("Parameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 80)

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append((step, value))

        if step is not None:
            self.logger.info(f"Step {step} - {name}: {value:.4f}")
        else:
            self.logger.info(f"{name}: {value:.4f}")

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/epoch number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_results(self, results: dict):
        """
        Log final experiment results.

        Args:
            results: Dictionary of result names and values
        """
        self.logger.info("=" * 80)
        self.logger.info("Final Results:")
        for key, value in results.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 80)

        # Log experiment duration
        duration = datetime.now() - self.start_time
        self.logger.info(f"Experiment duration: {duration}")

    def info(self, message: str):
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log an error message."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)


# Global logger instance
_default_logger = None


def get_logger() -> logging.Logger:
    """
    Get or create the default logger instance.

    Returns:
        Logger instance
    """
    global _default_logger

    if _default_logger is None:
        _default_logger = setup_logger()

    return _default_logger
