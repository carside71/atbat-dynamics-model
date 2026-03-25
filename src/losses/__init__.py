"""損失関数モジュール."""

from losses.focal import FocalLoss
from losses.multi_task import compute_loss
from losses.physics import PhysicsLoss

__all__ = ["FocalLoss", "PhysicsLoss", "compute_loss"]
