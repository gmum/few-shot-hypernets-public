from methods.hypernets.hypernet_poc import HyperNetPOC
from methods.hypernets.hypnettorch_wrapper import HypnettorchWrapper
from methods.hypernets.deprecated import HNPocAdaptTN, HNPocWithUniversalFinal, HyperNetConvFromDKT, HyperNetSepJoint, \
    HyperNetSupportConv, HyperNetSupportKernel, NoHNConditioning
from methods.hypernets.hypernet_kernel import HyperNetPocWithKernel, HyperNetPocSupportSupportKernel, \
    HNKernelBetweenSupportAndQuery, NoHNKernelBetweenSupportAndQuery

hypernet_types = {
    "hn_poc": HyperNetPOC,
    "hn_poc_kernel": HyperNetPocWithKernel,
    "hn_poc_sup_sup_kernel": HyperNetPocSupportSupportKernel,
    "hn_sup_kernel": HNKernelBetweenSupportAndQuery,
    "no_hn_sup_kernel": NoHNKernelBetweenSupportAndQuery,
    "hn_lib": HypnettorchWrapper,
    "_hn_uni_final": HNPocWithUniversalFinal,
    "_hn_poc_adapt_tn_val": HNPocAdaptTN,
    "_hn_from_dkt": HyperNetConvFromDKT,
    "_hn_sep_joint": HyperNetSepJoint,
    "_hn_cnv": HyperNetSupportConv,
    "_hn_kernel": HyperNetSupportKernel,
    "_no_hn_cond": NoHNConditioning,
}