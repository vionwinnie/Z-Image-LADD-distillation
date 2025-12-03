"""Helper utilities for Z-Image."""
import torch
from loguru import logger
from config import BYTES_PER_GB


def format_bytes(size: float) -> str:
    """
    Format bytes to GB string.
    
    Args:
        size: Size in bytes
        
    Returns:
        Formatted string in GB
    """
    n = size / BYTES_PER_GB
    return f"{n:.2f} GB"


def print_memory_stats(stage: str) -> None:
    """
    Print CUDA memory statistics.
    
    Args:
        stage: Description of current stage
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping memory stats")
        return
        
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    current_allocated = torch.cuda.memory_allocated()
    current_reserved = torch.cuda.memory_reserved()
    
    logger.info(f"[{stage}] Memory Stats:")
    logger.info(f"  Current Allocated: {format_bytes(current_allocated)}")
    logger.info(f"  Current Reserved:  {format_bytes(current_reserved)}")
    logger.info(f"  Peak Allocated:    {format_bytes(allocated)}")
    logger.info(f"  Peak Reserved:     {format_bytes(reserved)}")

