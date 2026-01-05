"""
resources.py

Handles individual resource nodes with intensity and depletion.
"""

import numpy as np

class ResourceField:
    """
    A resource node that is finite and vanishes after several uses.
    """
    def __init__(self, x, y, max_uses=5, intensity_per_use=1.0):
        self.x = x
        self.y = y
        self.max_uses = max_uses
        self.remaining_uses = max_uses
        self.intensity_per_use = intensity_per_use
        self.is_exhausted = False

    @property
    def intensity(self):
        """Map remaining uses to a float intensity for compatibility."""
        return float(self.remaining_uses)

    @property
    def max_intensity(self):
        return float(self.max_uses)

    def collect(self, amount=None):
        """
        Collect resource from the field. 
        In this finite model, each collection uses up 1 charge regardless of amount.
        """
        if self.remaining_uses <= 0:
            return 0.0
        
        self.remaining_uses -= 1
        collected = self.intensity_per_use
        
        if self.remaining_uses <= 0:
            self.is_exhausted = True
            
        return collected

    def regenerate(self, multiplier=1.0):
        """Finite resources do not regenerate."""
        pass

    def __repr__(self):
        return f"ResField({self.x},{self.y}, Uses={self.remaining_uses}/{self.max_uses})"
