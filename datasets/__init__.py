from .base import BaseLoader
from .adult import AdultLoader
from .celeba import CelebALoader
from .compas import CompasLoader
from .utkface import UTKFaceLoader

__all__ = ["celeba", "adult", "compas", "utkface"]