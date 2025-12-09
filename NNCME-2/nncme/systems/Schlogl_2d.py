import numpy as np
import torch

class Schlogl_2d:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.L = kwargs['L']
        print('Sites:',kwargs['Sites'])
        sites_str = kwargs['Sites']  # 例如 '2,4'
        rows, cols = map(int, sites_str.split(','))
        self.N_rows, self.N_cols = rows, cols
        self.Sites = rows * cols
        self.M = kwargs['M']
        self.bits = kwargs['bits']
        self.device = kwargs['device']
        self.MConstrain = kwargs['MConstrain']
        self.Para = kwargs['Para']
        self.IniDistri = kwargs['IniDistri']
        self.binary = kwargs['binary']
        self.absorb_state = kwargs['absorb_state']

    def site_index(self, i, j):
        return i * self.N_cols + j

    def Propensity(self, Win, Wout, cc, *args):
        Propensity_in = torch.prod(Win, 1) * cc
        Propensity_out = torch.prod(Wout, 1) * cc
        return Propensity_in, Propensity_out

    def rates(self):
        L = 1
        K = 4
        Sites = self.Sites
        L_total = L * Sites
        K_total = K * Sites

        r = torch.zeros(4)
        na = 1
        nb = self.Para
        c1, c2, c3, c4 = 2.676, 0.040, 108.102, 37.881
        d = 8.2207

        r[0] = c1 * na
        r[1] = c2
        r[2] = c3 * nb
        r[3] = c4
        r = r.repeat(Sites)

        # 初始分布
        D = 50
        initialD = np.tile(D, Sites).reshape(1, self.L)

        # 构造 update1
        Left = np.array([2, 3, 0, 1])
        Right = np.array([3, 2, 1, 0])
        update1L = np.zeros((L_total, K_total), dtype=int)
        update1R = np.zeros((L_total, K_total), dtype=int)
        for i in range(Sites):
            update1L[i * L:(i + 1) * L, i * K:(i + 1) * K] = Left
            update1R[i * L:(i + 1) * L, i * K:(i + 1) * K] = Right

        # 构造二维扩散 update2
        update2L_list = []
        update2R_list = []
        for i in range(self.N_rows):
            for j in range(self.N_cols):
                idx = self.site_index(i, j) * L
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < self.N_rows and 0 <= nj < self.N_cols:
                        nidx = self.site_index(ni, nj) * L
                        vecL = np.zeros(L_total, dtype=int)
                        vecR = np.zeros(L_total, dtype=int)
                        vecL[idx] = 1
                        vecR[nidx] = 1
                        update2L_list.append(vecL)
                        update2R_list.append(vecR)

        if d != 0:
            update2L = np.array(update2L_list).T
            update2R = np.array(update2R_list).T
            updateL = np.concatenate((update1L, update2L), axis=1)
            updateR = np.concatenate((update1R, update2R), axis=1)
            r = torch.cat((r, d * torch.ones(update2L.shape[1])))
        else:
            updateL = update1L
            updateR = update1R

        ReactionMatLeftTotal = torch.tensor(updateL).to(self.device)
        ReactionMatRightTotal = torch.tensor(updateR).to(self.device)

        MConstrain = np.zeros(1, dtype=int)
        conservation = np.ones(1, dtype=int)
        if self.binary:
            self.bits = int(np.ceil(np.log2(conservation)))

        return self.IniDistri, initialD, r, ReactionMatLeftTotal, ReactionMatRightTotal, MConstrain, conservation
