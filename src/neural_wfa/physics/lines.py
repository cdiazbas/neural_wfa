import numpy as np

class LineInfo:
    """
    Class to store the atomic data of spectral lines for WFA routines.
    """
    def __init__(self, cw=8542, verbose=False):
        self.larm = 4.668645048281451e-13

        if cw == 8542:
            self.j1 = 2.5
            self.j2 = 1.5
            self.g1 = 1.2
            self.g2 = 1.33
            self.cw = 8542.091
        elif cw == 6301:
            self.j1 = 2.0
            self.j2 = 2.0
            self.g1 = 1.84
            self.g2 = 1.50
            self.cw = 6301.4995
        elif cw == 6302:
            self.j1 = 1.0
            self.j2 = 0.0
            self.g1 = 2.49
            self.g2 = 0.0
            self.cw = 6302.4931
        elif cw == 8468:
            self.j1 = 1.0
            self.j2 = 1.0
            self.g1 = 2.50
            self.g2 = 2.49
            self.cw = 8468.4059
        elif cw == 6173:
            self.j1 = 1.0
            self.j2 = 0.0
            self.g1 = 2.50
            self.g2 = 0.0
            self.cw = 6173.3340
        elif cw == 5173:
            self.j1 = 1.0
            self.j2 = 1.0
            self.g1 = 1.50
            self.g2 = 2.0
            self.cw = 5172.6843
        elif cw == 5896:
            self.j1 = 0.5
            self.j2 = 0.5
            self.g1 = 2.00
            self.g2 = 2.0 / 3.0
            self.cw = 5895.9242
        else:
            if verbose:
                print(f"LineInfo: Warning, line {cw} not implemented, using zeros.")
            self.j1 = 0.0
            self.j2 = 0.0
            self.g1 = 0.0
            self.g2 = 0.0
            self.cw = float(cw) # Assuming cw is passed, maybe it's custom? 
            # In original code, it returned early. 
            # Here we let it proceed to calc geff/Gg (which will be 0) or handle graceful failure?
            # Original code returns early if not implemented.
            # But let's calculate geff/Gg as zeros safely.
            
        # Calculate parameters
        self._calculate_parameters(verbose)

    def _calculate_parameters(self, verbose=False):
        j1 = self.j1
        j2 = self.j2
        g1 = self.g1
        g2 = self.g2

        d = j1 * (j1 + 1.0) - j2 * (j2 + 1.0)
        self.geff = 0.5 * (g1 + g2) + 0.25 * (g1 - g2) * d
        ss = j1 * (j1 + 1.0) + j2 * (j2 + 1.0)
        dd = d
        gd = g1 - g2
        self.Gg = (self.geff * self.geff) - (
            0.0125 * gd * gd * (16.0 * ss - 7.0 * dd * dd - 4.0)
        )

        if verbose:
            print(
                "LineInfo: cw={0}, geff={1}, Gg={2}".format(
                    self.cw, self.geff, self.Gg
                )
            )
