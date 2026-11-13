"""Run folders by label, shared by the scripts that go back to the raw logs."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RUNS = {
    "baseline": "20261101_082229_three_tier_wc_custom_L1nat_H70",
    "base_rep": "20261105_075438_three_tier_wc_custom_L1nat_H70",
    "hintfree": "20261101_122705_three_tier_wc_dl_poll0_custom_L1nat_H70",
    "hintfree_ctl": "20261111_045808_three_tier_wc_dl_poll0_custom_L1nat_H70",
    "jit_p50": "20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "jit_p50_ctl": "20261111_024927_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "jit_p50_d_cut": "20261111_151101_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "jit_exact": "20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "ec_r13": "20261111_070012_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "ec_r1": "20261111_110558_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "nofc": "20261111_090311_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "nofc_noend": "20261111_130801_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "ORDER": "20261101_102434_three_tier_wc_oracle_v2_nu_custom_L1nat_H70",
}
CURRENT = open(os.path.join(RES, ".current_run")).read().strip()
