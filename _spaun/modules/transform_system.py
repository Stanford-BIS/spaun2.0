import numpy as np
from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from .._spa import Compare
from ..config import cfg
from ..utils import strs_to_inds, invol_matrix
from ..vocabs import vocab, ps_state_sp_strs, ps_dec_sp_strs
from .transform import Assoc_Mem_Transforms_Network


class TransformationSystem(Module):
    def __init__(self):
        super(TransformationSystem, self).__init__()
        self.init_module()

    @with_self
    def init_module(self):
        # ----- Input and output selectors ----- #
        self.select_in_a = cfg.make_selector(3)
        self.select_in_b = cfg.make_selector(6, represent_identity=True)
        self.select_out = cfg.make_selector(5, represent_identity=True)

        # ----- Mem inputs and outputs ----- #
        self.frm_mb1 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mb2 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mb3 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mbave = nengo.Node(size_in=cfg.sp_dim)

        nengo.Connection(self.frm_mb1, self.select_in_a.input0, synapse=None)
        nengo.Connection(self.frm_mb2, self.select_in_a.input1, synapse=None)
        nengo.Connection(self.frm_mb3, self.select_in_a.input2, synapse=None)

        nengo.Connection(self.frm_mb2, self.select_in_b.input2, synapse=None,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.frm_mb3, self.select_in_b.input3, synapse=None,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.frm_mbave, self.select_in_b.input4, synapse=None)
        nengo.Connection(self.frm_mb3, self.select_in_b.input5, synapse=None)

        nengo.Connection(self.frm_mb1, self.select_out.input2)

        # ----- Normalization networks for inputs to CConv and Compare ----- #
        self.norm_a = cfg.make_norm_net()
        self.norm_b = cfg.make_norm_net()

        nengo.Connection(self.select_in_a.output, self.norm_a.input)
        nengo.Connection(self.select_in_b.output, self.norm_b.input)

        # ----- Cir conv 1 Inputs (bypassable normalization) ----- #
        self.select_cconv1_a = cfg.make_selector(2)
        self.select_cconv1_b = cfg.make_selector(2, represent_identity=True)

        nengo.Connection(self.select_in_a.output, self.select_cconv1_a.input0)
        nengo.Connection(self.norm_a.output, self.select_cconv1_a.input1)

        nengo.Connection(self.select_in_b.output, self.select_cconv1_b.input0)
        nengo.Connection(self.norm_b.output, self.select_cconv1_b.input1)

        # ----- Cir conv 1 ----- #
        self.cconv1 = cfg.make_cir_conv(input_magnitude=cfg.trans_cconv_radius)

        nengo.Connection(self.select_cconv1_a.output, self.cconv1.A)
        nengo.Connection(self.select_cconv1_b.output, self.cconv1.B)
        nengo.Connection(self.cconv1.output, self.select_out.input3,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.cconv1.output, self.select_out.input4)

        # ----- Assoc memory transforms (for QA task) -----
        self.am_trfms = Assoc_Mem_Transforms_Network()

        nengo.Connection(self.frm_mb1, self.am_trfms.frm_mb1, synapse=None)
        nengo.Connection(self.frm_mb2, self.am_trfms.frm_mb2, synapse=None)
        nengo.Connection(self.frm_mb3, self.am_trfms.frm_mb3, synapse=None)

        nengo.Connection(self.cconv1.output, self.am_trfms.frm_cconv)

        nengo.Connection(self.am_trfms.pos1_to_pos, self.select_in_b.input0,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.am_trfms.pos1_to_num, self.select_in_b.input1,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.am_trfms.num_to_pos1, self.select_out.input0)
        nengo.Connection(self.am_trfms.pos_to_pos1, self.select_out.input1)

        # ----- Compare transformation (for counting task) -----
        self.compare = \
            Compare(vocab, output_no_match=True, threshold_outputs=0.5,
                    dot_product_input_magnitude=cfg.get_optimal_sp_radius(),
                    label="Compare")

        nengo.Connection(self.norm_a.output, self.compare.inputA,
                         transform=1.5)
        nengo.Connection(self.norm_b.output, self.compare.inputB,
                         transform=1.5)

        # ----- Output node -----
        self.output = nengo.Node(size_in=cfg.sp_dim)
        nengo.Connection(self.select_out.output, self.output, synapse=None)

        # ----- Set up module vocab inputs and outputs -----
        self.outputs = dict(compare=(self.compare.output, vocab))

    @with_self
    def setup_connections(self, parent_net):
        p_net = parent_net

        # Set up connections from ps module
        if hasattr(p_net, 'ps'):
            bias_node = nengo.Node(1)

            ps_state_mb_utils = p_net.ps.ps_state_utilities
            ps_dec_mb_utils = p_net.ps.ps_dec_utilities

            # Select IN A
            # - sel0 (MB1): State = QAP + QAK + TRANS1
            # - sel1 (MB2): State = TRANS2, CNT1
            # - sel2 (MB3): State = TRANS0
            in_a_sel0_inds = strs_to_inds(['QAP', 'QAK', 'TRANS1'],
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_a_sel0_inds],
                             self.select_in_a.sel0,
                             transform=[[1] * len(in_a_sel0_inds)])

            in_a_sel1_inds = strs_to_inds(['TRANS2', 'CNT1'],
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_a_sel1_inds],
                             self.select_in_a.sel1,
                             transform=[[1] * len(in_a_sel1_inds)])

            in_a_sel2_inds = strs_to_inds(['TRANS0'],
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_a_sel2_inds],
                             self.select_in_a.sel2)

            # Select IN B
            # - sel0 (~AM_P1): State = QAP
            # - sel1 (~AM_N1): State = QAK
            # - sel2 (~MB1): State = TRANS1 & Dec = -DECI
            # - sel3 (~MB2): State = TRANS2 & Dec = -DECI
            # - sel4 (MBAve): Dec = DECI
            # - sel5 (MB3): State = CNT1
            in_b_sel0_inds = strs_to_inds(['QAP'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_b_sel0_inds],
                             self.select_in_b.sel0)

            in_b_sel1_inds = strs_to_inds(['QAK'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_b_sel1_inds],
                             self.select_in_b.sel1)

            in_b_sel2 = cfg.make_thresh_ens_net()
            in_b_sel2_inds = strs_to_inds(['TRANS1'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_b_sel2_inds],
                             in_b_sel2.input)
            in_b_sel2_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[in_b_sel2_inds],
                             in_b_sel2.input, transform=-1)
            nengo.Connection(in_b_sel2.output, self.select_in_b.sel2)

            in_b_sel3 = cfg.make_thresh_ens_net()
            in_b_sel3_inds = strs_to_inds(['TRANS2'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_b_sel3_inds],
                             in_b_sel3.input)
            in_b_sel3_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[in_b_sel3_inds],
                             in_b_sel3.input, transform=-1)
            nengo.Connection(in_b_sel3.output, self.select_in_b.sel3)

            in_b_sel4_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[in_b_sel4_inds],
                             self.select_in_b.sel4)

            in_b_sel5_inds = strs_to_inds(['CNT1'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_b_sel5_inds],
                             self.select_in_b.sel5)

            # Select CCONV1 A
            # - sel0 (IN A): Dec = DECI
            # - sel1 (norm(IN A)): Dec = 1 -DECI
            cconv1_a_sel0_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[cconv1_a_sel0_inds],
                             self.select_cconv1_a.sel0)

            cconv1_a_sel1 = cfg.make_thresh_ens_net()
            cconv1_a_sel1_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(bias_node, cconv1_a_sel1.input)
            nengo.Connection(ps_dec_mb_utils[cconv1_a_sel1_inds],
                             cconv1_a_sel1.input, transform=-1)
            nengo.Connection(cconv1_a_sel1.output, self.select_cconv1_a.sel1)

            # Select CCONV1 B
            # - sel0 (IN B): Dec = DECI
            # - sel1 (norm(IN B)): Dec = 1 -DECI
            cconv1_b_sel0_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[cconv1_b_sel0_inds],
                             self.select_cconv1_b.sel0)

            cconv1_b_sel1 = cfg.make_thresh_ens_net()
            cconv1_b_sel1_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(bias_node, cconv1_b_sel1.input)
            nengo.Connection(ps_dec_mb_utils[cconv1_b_sel1_inds],
                             cconv1_b_sel1.input, transform=-1)
            nengo.Connection(cconv1_b_sel1.output, self.select_cconv1_b.sel1)

            # Select Output
            # - sel0 (AM_N2): State = QAP
            # - sel1 (AM_P2): State = QAK
            # - sel2 (MB1): State = TRANS0 + CNT1 & Dec = -DECI
            # - sel3 (~CC1 Out): State = TRANS1 + TRANS2 & Dec = -DECI
            # - sel4 (CC1 Out): Dec = DECI
            out_sel0_inds = strs_to_inds(['QAP'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[out_sel0_inds],
                             self.select_out.sel0)

            out_sel1_inds = strs_to_inds(['QAK'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[out_sel1_inds],
                             self.select_out.sel1)

            out_sel2 = cfg.make_thresh_ens_net()
            out_sel2_inds = strs_to_inds(['TRANS0', 'CNT1'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[out_sel2_inds], out_sel2.input,
                             transform=[[1] * len(out_sel2_inds)])
            out_sel2_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[out_sel2_inds], out_sel2.input,
                             transform=-1)
            nengo.Connection(out_sel2.output, self.select_out.sel2)

            out_sel3 = cfg.make_thresh_ens_net()
            out_sel3_inds = strs_to_inds(['TRANS1', 'TRANS2'],
                                         ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[out_sel3_inds], out_sel3.input,
                             transform=[[1] * len(out_sel3_inds)])
            out_sel3_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[out_sel3_inds], out_sel3.input,
                             transform=-1)
            nengo.Connection(out_sel3.output, self.select_out.sel3)

            out_sel4_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[out_sel4_inds],
                             self.select_out.sel4)

            # Disable input normalization for Dec = DECI
            dis_norm_inds = strs_to_inds(['FWD', 'DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[dis_norm_inds],
                             self.norm_a.disable,
                             transform=[[1] * len(dis_norm_inds)])
            nengo.Connection(ps_dec_mb_utils[dis_norm_inds],
                             self.norm_b.disable,
                             transform=[[1] * len(dis_norm_inds)])
        else:
            warn("TransformationSystem Module - Cannot connect from 'ps'")

        # Set up connections from memory module
        if hasattr(p_net, 'mem'):
            nengo.Connection(p_net.mem.mb1, self.frm_mb1)
            nengo.Connection(p_net.mem.mb2, self.frm_mb2)
            nengo.Connection(p_net.mem.mb3, self.frm_mb3)
            nengo.Connection(p_net.mem.mbave, self.frm_mbave)
        else:
            warn("TransformationSystem Module - Cannot connect from 'mem'")


class TransformationSystemDummy(TransformationSystem):
    def __init__(self):
        super(TransformationSystemDummy, self).__init__()
        self.init_module()

    @with_self
    def init_module(self):
        self.select_in_a = cfg.make_selector(2, n_ensembles=1,
                                             ens_dimensions=cfg.sp_dim,
                                             n_neurons=cfg.sp_dim)
        self.select_in_b = cfg.make_selector(5, n_ensembles=1,
                                             ens_dimensions=cfg.sp_dim,
                                             n_neurons=cfg.sp_dim)
        self.select_out = cfg.make_selector(4, n_ensembles=1,
                                            ens_dimensions=cfg.sp_dim,
                                            n_neurons=cfg.sp_dim)

        # ----- Mem inputs and outputs ----- #
        self.frm_mb1 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mb2 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mb3 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mbave = nengo.Node(size_in=cfg.sp_dim)

        # ----- Compare network (for counting task) -----
        def cmp_func(x, cmp_vocab):
            vec_A = x[:cfg.sp_dim]
            vec_B = x[cfg.sp_dim:]
            if np.linalg.norm(vec_A) != 0:
                vec_A = vec_A / np.linalg.norm(vec_A)
            if np.linalg.norm(vec_B) != 0:
                vec_B = vec_B / np.linalg.norm(vec_B)
            dot_val = np.dot(vec_A, vec_B)
            conj_val = 1 - dot_val
            if dot_val > conj_val:
                return cmp_vocab.parse('MATCH').v
            else:
                return cmp_vocab.parse('NO_MATCH').v

        self.compare = \
            nengo.Node(size_in=cfg.sp_dim * 2,
                       output=lambda t, x: cmp_func(x, cmp_vocab=vocab))

        nengo.Connection(self.frm_mb2, self.compare[:cfg.sp_dim])
        nengo.Connection(self.frm_mb3, self.compare[cfg.sp_dim:])

        # ----- Output node -----
        self.output = self.frm_mb1
        self.outputs = dict(compare=(self.compare, vocab))
