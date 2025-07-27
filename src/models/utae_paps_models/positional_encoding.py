# import torch
# import torch.nn as nn


# class PositionalEncoder(nn.Module):
#     def __init__(self, d, T=1000, repeat=None, offset=0):
#         super(PositionalEncoder, self).__init__()
#         self.d = d
#         self.T = T
#         self.repeat = repeat
#         self.denom = torch.pow(
#             T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
#         )
#         self.updated_location = False

#     def forward(self, batch_positions):
#         if not self.updated_location:
#             self.denom = self.denom.to(batch_positions.device)
#             self.updated_location = True
#         sinusoid_table = (
#             batch_positions[:, :, None] / self.denom[None, None, :]
#         )  # B x T x C
#         sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
#         sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

#         if self.repeat is not None:
#             sinusoid_table = torch.cat(
#                 [sinusoid_table for _ in range(self.repeat)], dim=-1
#             )

#         return sinusoid_table
    



import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T_rel = T              # For relative position
        self.T_doy = 365            # For DOY, hardcoded
        self.repeat = repeat
        self.offset = offset
        self.updated_location = False

        # denom for relative encoding
        self.denom_rel = torch.pow(
            self.T_rel, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        # denom for DOY encoding
        self.denom_doy = torch.pow(
            self.T_doy, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )

    def _sinusoidal_encoding(self, positions: torch.Tensor, denom: torch.Tensor):
        sinusoid = positions[:, :, None] / denom[None, None, :]  # (B, T, d)
        sinusoid[:, :, 0::2] = torch.sin(sinusoid[:, :, 0::2])
        sinusoid[:, :, 1::2] = torch.cos(sinusoid[:, :, 1::2])
        return sinusoid

    def forward(self, batch_positions: torch.Tensor, doys: torch.Tensor = None):
        """
        Args:
            batch_positions: (B, T) – relative positions
            doys: (B, T) – DOYs (optional)
        
        Returns:
            Combined or single positional encoding depending on if doy is provided
        """
        if not self.updated_location:
            self.denom_rel = self.denom_rel.to(batch_positions.device)
            self.denom_doy = self.denom_doy.to(batch_positions.device)
            self.updated_location = True

        rel_enc = self._sinusoidal_encoding(batch_positions, self.denom_rel)  # (B, T, d)

        if doys is None:
            final = rel_enc
        else:
            doy_enc = self._sinusoidal_encoding(doys, self.denom_doy)   
            final = doy_enc + rel_enc      # (B, T, d)
            # final = torch.cat([rel_enc, doy_enc], dim=-1)  # (B, T, 2d)

        if self.repeat is not None:
            final = torch.cat([final for _ in range(self.repeat)], dim=-1)

        return final




