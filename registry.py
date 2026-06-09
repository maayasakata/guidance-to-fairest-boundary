from methods.gfb import GFB
from datasets import CelebALoader, AdultLoader, CompasLoader, UTKFaceLoader

method_registry = {
    "gfb": GFB,
}

dataset_registry = {
    "celeba": CelebALoader,
    "adult": AdultLoader,
    "compas": CompasLoader,
    "utkface": UTKFaceLoader
}