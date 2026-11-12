# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _mantissa_table():
    v0_0 = tl.full((1,), 16777216.0, tl.float32)
    v0_1 = tl.full((1,), 11184811.0, tl.float32)
    v0_2 = tl.full((1,), 13421773.0, tl.float32)
    v0_3 = tl.full((1,), 9586981.0, tl.float32)
    v0_4 = tl.full((1,), 14913081.0, tl.float32)
    v0_5 = tl.full((1,), 10324441.0, tl.float32)
    v0_6 = tl.full((1,), 12201612.0, tl.float32)
    v0_7 = tl.full((1,), 8947849.0, tl.float32)
    v0_8 = tl.full((1,), 15790321.0, tl.float32)
    v0_9 = tl.full((1,), 10737418.0, tl.float32)
    v0_10 = tl.full((1,), 12782641.0, tl.float32)
    v0_11 = tl.full((1,), 9256395.0, tl.float32)
    v0_12 = tl.full((1,), 14128182.0, tl.float32)
    v0_13 = tl.full((1,), 9942054.0, tl.float32)
    v0_14 = tl.full((1,), 11671107.0, tl.float32)
    v0_15 = tl.full((1,), 8659208.0, tl.float32)
    v0_16 = tl.full((1,), 16268816.0, tl.float32)
    v0_17 = tl.full((1,), 10956549.0, tl.float32)
    v0_18 = tl.full((1,), 13094412.0, tl.float32)
    v0_19 = tl.full((1,), 9418788.0, tl.float32)
    v0_20 = tl.full((1,), 14510025.0, tl.float32)
    v0_21 = tl.full((1,), 10129640.0, tl.float32)
    v0_22 = tl.full((1,), 11930465.0, tl.float32)
    v0_23 = tl.full((1,), 8801162.0, tl.float32)
    v0_24 = tl.full((1,), 15339169.0, tl.float32)
    v0_25 = tl.full((1,), 10526881.0, tl.float32)
    v0_26 = tl.full((1,), 12485370.0, tl.float32)
    v0_27 = tl.full((1,), 9099507.0, tl.float32)
    v0_28 = tl.full((1,), 13765921.0, tl.float32)
    v0_29 = tl.full((1,), 9761289.0, tl.float32)
    v0_30 = tl.full((1,), 11422785.0, tl.float32)
    v0_31 = tl.full((1,), 8521761.0, tl.float32)
    v0_32 = tl.full((1,), 16519105.0, tl.float32)
    v0_33 = tl.full((1,), 11069503.0, tl.float32)
    v0_34 = tl.full((1,), 13256072.0, tl.float32)
    v0_35 = tl.full((1,), 9502140.0, tl.float32)
    v0_36 = tl.full((1,), 14708792.0, tl.float32)
    v0_37 = tl.full((1,), 10226113.0, tl.float32)
    v0_38 = tl.full((1,), 12064515.0, tl.float32)
    v0_39 = tl.full((1,), 8873899.0, tl.float32)
    v0_40 = tl.full((1,), 15561476.0, tl.float32)
    v0_41 = tl.full((1,), 10631107.0, tl.float32)
    v0_42 = tl.full((1,), 12632257.0, tl.float32)
    v0_43 = tl.full((1,), 9177281.0, tl.float32)
    v0_44 = tl.full((1,), 13944699.0, tl.float32)
    v0_45 = tl.full((1,), 9850842.0, tl.float32)
    v0_46 = tl.full((1,), 11545611.0, tl.float32)
    v0_47 = tl.full((1,), 8589935.0, tl.float32)
    v0_48 = tl.full((1,), 16025997.0, tl.float32)
    v0_49 = tl.full((1,), 10845877.0, tl.float32)
    v0_50 = tl.full((1,), 12936648.0, tl.float32)
    v0_51 = tl.full((1,), 9336885.0, tl.float32)
    v0_52 = tl.full((1,), 14316558.0, tl.float32)
    v0_53 = tl.full((1,), 10034970.0, tl.float32)
    v0_54 = tl.full((1,), 11799361.0, tl.float32)
    v0_55 = tl.full((1,), 8729608.0, tl.float32)
    v0_56 = tl.full((1,), 15123124.0, tl.float32)
    v0_57 = tl.full((1,), 10424678.0, tl.float32)
    v0_58 = tl.full((1,), 12341860.0, tl.float32)
    v0_59 = tl.full((1,), 9023041.0, tl.float32)
    v0_60 = tl.full((1,), 13591669.0, tl.float32)
    v0_61 = tl.full((1,), 9673350.0, tl.float32)
    v0_62 = tl.full((1,), 11302546.0, tl.float32)
    v0_63 = tl.full((1,), 8454660.0, tl.float32)
    v0_64 = tl.full((1,), 16647160.0, tl.float32)
    v0_65 = tl.full((1,), 11126858.0, tl.float32)
    v0_66 = tl.full((1,), 13338408.0, tl.float32)
    v0_67 = tl.full((1,), 9544372.0, tl.float32)
    v0_68 = tl.full((1,), 14810232.0, tl.float32)
    v0_69 = tl.full((1,), 10275041.0, tl.float32)
    v0_70 = tl.full((1,), 12132676.0, tl.float32)
    v0_71 = tl.full((1,), 8910721.0, tl.float32)
    v0_72 = tl.full((1,), 15675063.0, tl.float32)
    v0_73 = tl.full((1,), 10683998.0, tl.float32)
    v0_74 = tl.full((1,), 12707004.0, tl.float32)
    v0_75 = tl.full((1,), 9216668.0, tl.float32)
    v0_76 = tl.full((1,), 14035841.0, tl.float32)
    v0_77 = tl.full((1,), 9896238.0, tl.float32)
    v0_78 = tl.full((1,), 11608020.0, tl.float32)
    v0_79 = tl.full((1,), 8624432.0, tl.float32)
    v0_80 = tl.full((1,), 16146494.0, tl.float32)
    v0_81 = tl.full((1,), 10900932.0, tl.float32)
    v0_82 = tl.full((1,), 13015052.0, tl.float32)
    v0_83 = tl.full((1,), 9377658.0, tl.float32)
    v0_84 = tl.full((1,), 14412642.0, tl.float32)
    v0_85 = tl.full((1,), 10082083.0, tl.float32)
    v0_86 = tl.full((1,), 11864551.0, tl.float32)
    v0_87 = tl.full((1,), 8765239.0, tl.float32)
    v0_88 = tl.full((1,), 15230380.0, tl.float32)
    v0_89 = tl.full((1,), 10475530.0, tl.float32)
    v0_90 = tl.full((1,), 12413200.0, tl.float32)
    v0_91 = tl.full((1,), 9061112.0, tl.float32)
    v0_92 = tl.full((1,), 13678240.0, tl.float32)
    v0_93 = tl.full((1,), 9717121.0, tl.float32)
    v0_94 = tl.full((1,), 11362347.0, tl.float32)
    v0_95 = tl.full((1,), 8488078.0, tl.float32)
    v0_96 = tl.full((1,), 16393005.0, tl.float32)
    v0_97 = tl.full((1,), 11012737.0, tl.float32)
    v0_98 = tl.full((1,), 13174746.0, tl.float32)
    v0_99 = tl.full((1,), 9460280.0, tl.float32)
    v0_100 = tl.full((1,), 14608732.0, tl.float32)
    v0_101 = tl.full((1,), 10177648.0, tl.float32)
    v0_102 = tl.full((1,), 11997115.0, tl.float32)
    v0_103 = tl.full((1,), 8837381.0, tl.float32)
    v0_104 = tl.full((1,), 15449523.0, tl.float32)
    v0_105 = tl.full((1,), 10578737.0, tl.float32)
    v0_106 = tl.full((1,), 12558384.0, tl.float32)
    v0_107 = tl.full((1,), 9138228.0, tl.float32)
    v0_108 = tl.full((1,), 13854733.0, tl.float32)
    v0_109 = tl.full((1,), 9805861.0, tl.float32)
    v0_110 = tl.full((1,), 11483870.0, tl.float32)
    v0_111 = tl.full((1,), 8555712.0, tl.float32)
    v0_112 = tl.full((1,), 15907286.0, tl.float32)
    v0_113 = tl.full((1,), 10791375.0, tl.float32)
    v0_114 = tl.full((1,), 12859184.0, tl.float32)
    v0_115 = tl.full((1,), 9296466.0, tl.float32)
    v0_116 = tl.full((1,), 14221746.0, tl.float32)
    v0_117 = tl.full((1,), 9988296.0, tl.float32)
    v0_118 = tl.full((1,), 11734883.0, tl.float32)
    v0_119 = tl.full((1,), 8694266.0, tl.float32)
    v0_120 = tl.full((1,), 15017368.0, tl.float32)
    v0_121 = tl.full((1,), 10374317.0, tl.float32)
    v0_122 = tl.full((1,), 12271335.0, tl.float32)
    v0_123 = tl.full((1,), 8985287.0, tl.float32)
    v0_124 = tl.full((1,), 13506186.0, tl.float32)
    v0_125 = tl.full((1,), 9629972.0, tl.float32)
    v0_126 = tl.full((1,), 11243370.0, tl.float32)
    v0_127 = tl.full((1,), 8421505.0, tl.float32)
    v1_0 = tl.join(v0_0, v0_1).reshape((2,))
    v1_1 = tl.join(v0_2, v0_3).reshape((2,))
    v1_2 = tl.join(v0_4, v0_5).reshape((2,))
    v1_3 = tl.join(v0_6, v0_7).reshape((2,))
    v1_4 = tl.join(v0_8, v0_9).reshape((2,))
    v1_5 = tl.join(v0_10, v0_11).reshape((2,))
    v1_6 = tl.join(v0_12, v0_13).reshape((2,))
    v1_7 = tl.join(v0_14, v0_15).reshape((2,))
    v1_8 = tl.join(v0_16, v0_17).reshape((2,))
    v1_9 = tl.join(v0_18, v0_19).reshape((2,))
    v1_10 = tl.join(v0_20, v0_21).reshape((2,))
    v1_11 = tl.join(v0_22, v0_23).reshape((2,))
    v1_12 = tl.join(v0_24, v0_25).reshape((2,))
    v1_13 = tl.join(v0_26, v0_27).reshape((2,))
    v1_14 = tl.join(v0_28, v0_29).reshape((2,))
    v1_15 = tl.join(v0_30, v0_31).reshape((2,))
    v1_16 = tl.join(v0_32, v0_33).reshape((2,))
    v1_17 = tl.join(v0_34, v0_35).reshape((2,))
    v1_18 = tl.join(v0_36, v0_37).reshape((2,))
    v1_19 = tl.join(v0_38, v0_39).reshape((2,))
    v1_20 = tl.join(v0_40, v0_41).reshape((2,))
    v1_21 = tl.join(v0_42, v0_43).reshape((2,))
    v1_22 = tl.join(v0_44, v0_45).reshape((2,))
    v1_23 = tl.join(v0_46, v0_47).reshape((2,))
    v1_24 = tl.join(v0_48, v0_49).reshape((2,))
    v1_25 = tl.join(v0_50, v0_51).reshape((2,))
    v1_26 = tl.join(v0_52, v0_53).reshape((2,))
    v1_27 = tl.join(v0_54, v0_55).reshape((2,))
    v1_28 = tl.join(v0_56, v0_57).reshape((2,))
    v1_29 = tl.join(v0_58, v0_59).reshape((2,))
    v1_30 = tl.join(v0_60, v0_61).reshape((2,))
    v1_31 = tl.join(v0_62, v0_63).reshape((2,))
    v1_32 = tl.join(v0_64, v0_65).reshape((2,))
    v1_33 = tl.join(v0_66, v0_67).reshape((2,))
    v1_34 = tl.join(v0_68, v0_69).reshape((2,))
    v1_35 = tl.join(v0_70, v0_71).reshape((2,))
    v1_36 = tl.join(v0_72, v0_73).reshape((2,))
    v1_37 = tl.join(v0_74, v0_75).reshape((2,))
    v1_38 = tl.join(v0_76, v0_77).reshape((2,))
    v1_39 = tl.join(v0_78, v0_79).reshape((2,))
    v1_40 = tl.join(v0_80, v0_81).reshape((2,))
    v1_41 = tl.join(v0_82, v0_83).reshape((2,))
    v1_42 = tl.join(v0_84, v0_85).reshape((2,))
    v1_43 = tl.join(v0_86, v0_87).reshape((2,))
    v1_44 = tl.join(v0_88, v0_89).reshape((2,))
    v1_45 = tl.join(v0_90, v0_91).reshape((2,))
    v1_46 = tl.join(v0_92, v0_93).reshape((2,))
    v1_47 = tl.join(v0_94, v0_95).reshape((2,))
    v1_48 = tl.join(v0_96, v0_97).reshape((2,))
    v1_49 = tl.join(v0_98, v0_99).reshape((2,))
    v1_50 = tl.join(v0_100, v0_101).reshape((2,))
    v1_51 = tl.join(v0_102, v0_103).reshape((2,))
    v1_52 = tl.join(v0_104, v0_105).reshape((2,))
    v1_53 = tl.join(v0_106, v0_107).reshape((2,))
    v1_54 = tl.join(v0_108, v0_109).reshape((2,))
    v1_55 = tl.join(v0_110, v0_111).reshape((2,))
    v1_56 = tl.join(v0_112, v0_113).reshape((2,))
    v1_57 = tl.join(v0_114, v0_115).reshape((2,))
    v1_58 = tl.join(v0_116, v0_117).reshape((2,))
    v1_59 = tl.join(v0_118, v0_119).reshape((2,))
    v1_60 = tl.join(v0_120, v0_121).reshape((2,))
    v1_61 = tl.join(v0_122, v0_123).reshape((2,))
    v1_62 = tl.join(v0_124, v0_125).reshape((2,))
    v1_63 = tl.join(v0_126, v0_127).reshape((2,))
    v2_0 = tl.join(v1_0, v1_1).reshape((4,))
    v2_1 = tl.join(v1_2, v1_3).reshape((4,))
    v2_2 = tl.join(v1_4, v1_5).reshape((4,))
    v2_3 = tl.join(v1_6, v1_7).reshape((4,))
    v2_4 = tl.join(v1_8, v1_9).reshape((4,))
    v2_5 = tl.join(v1_10, v1_11).reshape((4,))
    v2_6 = tl.join(v1_12, v1_13).reshape((4,))
    v2_7 = tl.join(v1_14, v1_15).reshape((4,))
    v2_8 = tl.join(v1_16, v1_17).reshape((4,))
    v2_9 = tl.join(v1_18, v1_19).reshape((4,))
    v2_10 = tl.join(v1_20, v1_21).reshape((4,))
    v2_11 = tl.join(v1_22, v1_23).reshape((4,))
    v2_12 = tl.join(v1_24, v1_25).reshape((4,))
    v2_13 = tl.join(v1_26, v1_27).reshape((4,))
    v2_14 = tl.join(v1_28, v1_29).reshape((4,))
    v2_15 = tl.join(v1_30, v1_31).reshape((4,))
    v2_16 = tl.join(v1_32, v1_33).reshape((4,))
    v2_17 = tl.join(v1_34, v1_35).reshape((4,))
    v2_18 = tl.join(v1_36, v1_37).reshape((4,))
    v2_19 = tl.join(v1_38, v1_39).reshape((4,))
    v2_20 = tl.join(v1_40, v1_41).reshape((4,))
    v2_21 = tl.join(v1_42, v1_43).reshape((4,))
    v2_22 = tl.join(v1_44, v1_45).reshape((4,))
    v2_23 = tl.join(v1_46, v1_47).reshape((4,))
    v2_24 = tl.join(v1_48, v1_49).reshape((4,))
    v2_25 = tl.join(v1_50, v1_51).reshape((4,))
    v2_26 = tl.join(v1_52, v1_53).reshape((4,))
    v2_27 = tl.join(v1_54, v1_55).reshape((4,))
    v2_28 = tl.join(v1_56, v1_57).reshape((4,))
    v2_29 = tl.join(v1_58, v1_59).reshape((4,))
    v2_30 = tl.join(v1_60, v1_61).reshape((4,))
    v2_31 = tl.join(v1_62, v1_63).reshape((4,))
    v3_0 = tl.join(v2_0, v2_1).reshape((8,))
    v3_1 = tl.join(v2_2, v2_3).reshape((8,))
    v3_2 = tl.join(v2_4, v2_5).reshape((8,))
    v3_3 = tl.join(v2_6, v2_7).reshape((8,))
    v3_4 = tl.join(v2_8, v2_9).reshape((8,))
    v3_5 = tl.join(v2_10, v2_11).reshape((8,))
    v3_6 = tl.join(v2_12, v2_13).reshape((8,))
    v3_7 = tl.join(v2_14, v2_15).reshape((8,))
    v3_8 = tl.join(v2_16, v2_17).reshape((8,))
    v3_9 = tl.join(v2_18, v2_19).reshape((8,))
    v3_10 = tl.join(v2_20, v2_21).reshape((8,))
    v3_11 = tl.join(v2_22, v2_23).reshape((8,))
    v3_12 = tl.join(v2_24, v2_25).reshape((8,))
    v3_13 = tl.join(v2_26, v2_27).reshape((8,))
    v3_14 = tl.join(v2_28, v2_29).reshape((8,))
    v3_15 = tl.join(v2_30, v2_31).reshape((8,))
    v4_0 = tl.join(v3_0, v3_1).reshape((16,))
    v4_1 = tl.join(v3_2, v3_3).reshape((16,))
    v4_2 = tl.join(v3_4, v3_5).reshape((16,))
    v4_3 = tl.join(v3_6, v3_7).reshape((16,))
    v4_4 = tl.join(v3_8, v3_9).reshape((16,))
    v4_5 = tl.join(v3_10, v3_11).reshape((16,))
    v4_6 = tl.join(v3_12, v3_13).reshape((16,))
    v4_7 = tl.join(v3_14, v3_15).reshape((16,))
    v5_0 = tl.join(v4_0, v4_1).reshape((32,))
    v5_1 = tl.join(v4_2, v4_3).reshape((32,))
    v5_2 = tl.join(v4_4, v4_5).reshape((32,))
    v5_3 = tl.join(v4_6, v4_7).reshape((32,))
    v6_0 = tl.join(v5_0, v5_1).reshape((64,))
    v6_1 = tl.join(v5_2, v5_3).reshape((64,))
    v7_0 = tl.join(v6_0, v6_1).reshape((128,))
    return v7_0


@triton.jit
def _computed_table():
    index = tl.arange(0, 128)
    divisor = index + 128
    normalized = ((index << 16) + 1065353216).to(tl.float32, bitcast=True)
    quotient = (tl.div_rn(1.0, normalized) * 8388608.0).to(tl.int32)
    remainder = 1073741824 - quotient * divisor
    below = remainder < 0
    quotient = quotient - below.to(tl.int32)
    remainder = remainder + tl.where(below, divisor, 0)
    above = remainder >= divisor
    quotient = quotient + above.to(tl.int32)
    remainder = remainder - tl.where(above, divisor, 0)
    q = (
        2 * quotient
        + (4 * remainder > divisor).to(tl.int32)
        + (4 * remainder >= 3 * divisor).to(tl.int32)
    )
    return q.to(tl.float32)


@triton.jit
def _full_inverse(amax):
    bits = amax.to(tl.int32, bitcast=True)
    exponent = bits >> 23
    divisor = (bits >> 16 & 127) + 128
    normalized = ((divisor - 128 << 16) + 1065353216).to(
        tl.float32, bitcast=True
    )
    estimate = tl.div_rn(1.0, normalized) * 8388608.0
    quotient = estimate.to(tl.int32)
    remainder = 1073741824 - quotient * divisor
    below = remainder < 0
    quotient = quotient - below.to(tl.int32)
    remainder = remainder + tl.where(below, divisor, 0)
    above = remainder >= divisor
    quotient = quotient + above.to(tl.int32)
    remainder = remainder - tl.where(above, divisor, 0)
    normal = exponent <= 252
    twice = remainder * 2
    normal_q = quotient * 2 + (twice >= divisor).to(tl.int32)
    normal_r = twice - tl.where(twice >= divisor, divisor, 0)
    last = exponent == 254
    high_q = tl.where(last, quotient >> 1, quotient)
    high_r = tl.where(last, remainder + (quotient & 1) * divisor, remainder)
    high_d = tl.where(last, divisor * 2, divisor)
    q = tl.where(normal, normal_q, high_q)
    r = tl.where(normal, normal_r, high_r)
    d = tl.where(normal, divisor, high_d)
    q += ((2 * r > d) | (2 * r == d) & (q & 1 != 0)).to(tl.int32)
    shift = tl.where(q >= 8388608, 0, tl.where(q >= 4194304, 1, 2))
    high_bits = 9 - shift << 23 | q << shift & 8388607
    normal_bits = q + (252 - tl.minimum(exponent, 252) << 23)
    reciprocal = tl.where(normal, normal_bits, high_bits).to(
        tl.float32, bitcast=True
    )
    inv = reciprocal * tl.where(normal, 448.0, 1.75)
    inv = tl.where(bits < 953286656, 4480000.0, inv)
    inv = tl.where(bits == 2139095040, 0.0, inv)
    return tl.where(bits > 2139095040, float("nan"), inv)


@triton.jit
def _reference_inverse(amax, table):
    bits = amax.to(tl.int32, bitcast=True)
    exponent = bits >> 23
    index = bits >> 16 & 127
    q = tl.gather(table, index, 0).to(tl.int32)
    normal_bits = q + (252 - tl.minimum(exponent, 252) << 23)
    inv = normal_bits.to(tl.float32, bitcast=True) * 448.0
    inv = tl.where(bits < 953286656, 4480000.0, inv)
    if tl.max(exponent.to(tl.float32), 0) > 252.0:
        inv = tl.where(exponent > 252, _full_inverse(amax), inv)
    return inv


@triton.jit
def _encode(v):
    raw = v.to(tl.int32, bitcast=True)
    mag = raw & 2147483647
    exponent = tl.maximum(mag >> 23, 121)
    bias = exponent * 8388616 + 167771192
    rounded = (
        mag.to(tl.float32, bitcast=True) + bias.to(tl.float32, bitcast=True)
    ).to(tl.int32, bitcast=True)
    code = rounded & 255 | raw >> 24 & 128
    return tl.where(mag > 2139095040, 127, code).to(tl.uint8)


@triton.jit
def _expert(
    A,
    Q,
    S,
    OFF,
    PS,
    KG: tl.constexpr,
    G: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
):
    e = tl.program_id(0)
    start = tl.load(OFF + e * OS)
    count = tl.load(PS + e * PS0)
    table = _mantissa_table()
    for tile in range(
        tl.program_id(1), tl.cdiv(count * KG, NG), tl.num_programs(1)
    ):
        ids = tile * NG + tl.arange(0, NG)
        cols = tl.arange(0, B)
        offsets = (start * KG + ids[:, None]) * G + cols[None, :]
        mask = (ids[:, None] < count * KG) & (cols[None, :] < G)
        x = tl.load(A + offsets, mask, 0).to(tl.float32)
        amax = tl.maximum(
            tl.max(
                (x.to(tl.uint32, bitcast=True) & 2147483647)
                .to(tl.int32)
                .to(tl.float32),
                1,
            ).to(tl.int32),
            953267991,
        ).to(tl.float32, bitcast=True)
        v = x * _reference_inverse(amax, table)[:, None]
        tl.store(Q + offsets, _encode(v), mask)
        dst = start * KG + ids % KG * count + ids // KG
        tl.store(S + dst, tl.div_rn(amax, 448.0), ids < count * KG)


@triton.jit
def _quant_range(
    A,
    Q,
    T,
    table,
    start,
    count,
    G: tl.constexpr,
    TOTAL: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
):
    for tile in range(
        tl.program_id(0), tl.cdiv(count, NG), tl.num_programs(0)
    ):
        ids = tile * NG + tl.arange(0, NG)
        cols = tl.arange(0, B)
        offsets = (start + ids[:, None]) * G + cols[None, :]
        x = tl.load(
            A + offsets, (ids[:, None] < count) & (cols[None, :] < G), 0
        ).to(tl.float32)
        amax = tl.maximum(
            tl.max(
                (x.to(tl.uint32, bitcast=True) & 2147483647)
                .to(tl.int32)
                .to(tl.float32),
                1,
            ).to(tl.int32),
            953267991,
        ).to(tl.float32, bitcast=True)
        v = x * _reference_inverse(amax, table)[:, None]
        tl.store(
            Q + offsets,
            _encode(v),
            (ids[:, None] < count) & (cols[None, :] < G),
        )
        tl.store(T + start + ids, tl.div_rn(amax, 448.0), ids < count)


@triton.jit
def _quant(
    A,
    Q,
    T,
    OFF,
    PS,
    KG: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    G: tl.constexpr,
    TOTAL: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
):
    if TOTAL >= 196608:
        table = _computed_table()
    else:
        table = _mantissa_table()
    experts = tl.arange(0, BE)
    expert_starts = tl.load(OFF + experts * OS, experts < E, 0)
    counts = tl.load(PS + experts * PS0, experts < E, 0)
    ends = tl.load(OFF + (experts + 1) * OS, experts + 1 < E, TOTAL // KG)
    contiguous = (
        tl.sum(
            ((experts < E) & (expert_starts + counts != ends)).to(tl.int32), 0
        )
        == 0
    )
    contiguous = contiguous & (tl.load(OFF) == 0)
    if contiguous:
        for tile in range(
            tl.program_id(0), tl.cdiv(TOTAL, NG), tl.num_programs(0)
        ):
            ids = tile * NG + tl.arange(0, NG)
            cols = tl.arange(0, B)
            offsets = ids[:, None] * G + cols[None, :]
            x = tl.load(
                A + offsets, (ids[:, None] < TOTAL) & (cols[None, :] < G), 0
            ).to(tl.float32)
            amax = tl.maximum(
                tl.max(
                    (x.to(tl.uint32, bitcast=True) & 2147483647)
                    .to(tl.int32)
                    .to(tl.float32),
                    1,
                ).to(tl.int32),
                953267991,
            ).to(tl.float32, bitcast=True)
            v = x * _reference_inverse(amax, table)[:, None]
            tl.store(
                Q + offsets,
                _encode(v),
                (ids[:, None] < TOTAL) & (cols[None, :] < G),
            )
            tl.store(T + ids, tl.div_rn(amax, 448.0), ids < TOTAL)
    else:
        for e in range(E):
            start = tl.load(OFF + e * OS) * KG
            count = tl.load(PS + e * PS0) * KG
            _quant_range(A, Q, T, table, start, count, G, TOTAL, NG, B)


@triton.jit
def _reorder(
    T,
    S,
    OFF,
    PS,
    KG: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    B: tl.constexpr,
):
    e = tl.program_id(0)
    start = tl.load(OFF + e * OS)
    count = tl.load(PS + e * PS0)
    tiles_g: tl.constexpr = tl.cdiv(KG, 16)
    for tile in range(
        tl.program_id(1), tl.cdiv(count, 16) * tiles_g, tl.num_programs(1)
    ):
        rows = tile // tiles_g * 16 + tl.arange(0, 16)
        groups = tile % tiles_g * 16 + tl.arange(0, 16)
        mask = (rows[:, None] < count) & (groups[None, :] < KG)
        value = tl.load(
            T + (start + rows[:, None]) * KG + groups[None, :], mask, 0
        )
        tl.store(
            S + start * KG + groups[None, :] * count + rows[:, None],
            value,
            mask,
        )


@triton.jit
def _partial(A, P, G: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    tile = tl.program_id(0)
    group = tile // C
    cols = tile % C * B + tl.arange(0, B)
    x = tl.load(A + group * G + cols, cols < G, 0).to(tl.float32)
    tl.store(
        P + tile,
        tl.max(
            (x.to(tl.uint32, bitcast=True) & 2147483647)
            .to(tl.int32)
            .to(tl.float32),
            0,
        )
        .to(tl.int32)
        .to(tl.float32, bitcast=True),
    )


@triton.jit
def _wide_stats(
    P,
    InvPtr,
    T,
    C: tl.constexpr,
    BC: tl.constexpr,
    TOTAL: tl.constexpr,
    NG: tl.constexpr,
):
    ids = tl.program_id(0) * NG + tl.arange(0, NG)
    pieces = tl.arange(0, BC)
    partial = tl.load(
        P + ids[:, None] * C + pieces[None, :],
        (ids[:, None] < TOTAL) & (pieces[None, :] < C),
        0,
    )
    maximum = tl.maximum(
        tl.max(partial.to(tl.int32, bitcast=True).to(tl.float32), 1).to(
            tl.int32
        ),
        953267991,
    ).to(tl.float32, bitcast=True)
    tl.store(
        InvPtr + ids,
        _reference_inverse(maximum, _mantissa_table()),
        ids < TOTAL,
    )
    tl.store(T + ids, tl.div_rn(maximum, 448.0), ids < TOTAL)


@triton.jit
def _wide_quant(
    A,
    Q,
    InvPtr,
    OFF,
    PS,
    KG: tl.constexpr,
    E: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    G: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
):
    tile = tl.program_id(0)
    group = tile // C
    cols = tile % C * B + tl.arange(0, B)
    row = group // KG
    lo = 0
    hi = E
    while lo < hi:
        mid = (lo + hi) // 2
        off = tl.load(OFF + mid * OS)
        if off <= row:
            lo = mid + 1
        else:
            hi = mid
    expert = tl.maximum(lo - 1, 0)
    off = tl.load(OFF + expert * OS)
    count = tl.load(PS + expert * PS0)
    if (lo > 0) & (row < off + count):
        inverse = tl.load(InvPtr + group)
        x = tl.load(A + group * G + cols, cols < G, 0).to(tl.float32)
        tl.store(Q + group * G + cols, _encode(x * inverse), cols < G)


def per_token_group_quant_fp8_moe_mn_major(
    A, expert_offsets, problem_sizes, group_size
):
    if (
        A.numel() >= 2147483648
        or problem_sizes.shape[0] > 65535
        or group_size > 4096
        or (
            group_size > 256
            and (A.numel() // group_size) * triton.cdiv(group_size, 128)
            > 65535
        )
    ):
        return _large_entry(A, expert_offsets, problem_sizes, group_size)
    m, k = A.shape
    kg = k // group_size
    e = problem_sizes.shape[0]
    q = torch.empty((m, k), dtype=torch.uint8, device=A.device)
    s = torch.empty((m, kg), dtype=torch.float32, device=A.device)
    block = triton.next_power_of_2(group_size)
    ng = min(32, max(1, (2048 if block > 128 else 4096) // block))
    if group_size == 128 and m * kg >= 98304:
        ng = 64
    if m * kg and e and (block > 256):
        chunks = triton.cdiv(group_size, 128)
        p = torch.empty((m * kg, chunks), dtype=torch.float32, device=A.device)
        t = torch.empty_like(s)
        _partial[m * kg * chunks,](A, p, group_size, chunks, 128, num_warps=4)
        inv = torch.empty_like(s)
        _wide_stats[triton.cdiv(m * kg, 8),](
            p,
            inv,
            t,
            chunks,
            triton.next_power_of_2(chunks),
            m * kg,
            8,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _wide_quant[m * kg * chunks,](
            A,
            q,
            inv,
            expert_offsets,
            problem_sizes,
            kg,
            e,
            expert_offsets.stride(0),
            problem_sizes.stride(0),
            group_size,
            chunks,
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _reorder[e, max(1, 40 // e)](
            t,
            s,
            expert_offsets,
            problem_sizes,
            kg,
            expert_offsets.stride(0),
            problem_sizes.stride(0),
            256,
            num_warps=4,
        )
    elif (
        group_size == 128
        and m * kg >= 196608
        and 0 < e <= 32
        and e * (40 // e) <= 32
    ):
        _pair_uniform[(40,)](
            A.view(torch.int32),
            q.view(torch.int16),
            s,
            expert_offsets,
            problem_sizes,
            e,
            triton.next_power_of_2(e),
            kg,
            expert_offsets.stride(0),
            problem_sizes.stride(0),
            64,
            num_warps=4,
            enable_fp_fusion=False,
            enable_ubuf_saving=True,
        )
    elif group_size == 128 and m * kg and e:
        saving = m * kg >= 98304 and e * max(1, 40 // e) <= 32
        compact = kg > 24 and kg % 8 == 0 and kg % 64 != 0 and not saving
        _pair_rows[(e, min(max(1, 40 // e), triton.cdiv(m, 64) * kg))](
            A.view(torch.int32),
            q.view(torch.int16),
            s,
            expert_offsets,
            problem_sizes,
            kg,
            expert_offsets.stride(0),
            problem_sizes.stride(0),
            64,
            compact,
            num_warps=4,
            enable_fp_fusion=False,
            enable_ubuf_saving=(saving or compact),
        )
    elif m * kg >= 65536 and e:
        t = torch.empty_like(s)
        kernel = (
            _quant_paired if group_size == 128 and m * kg >= 98304 else _quant
        )
        packed_path = group_size == 128 and m * kg >= 98304
        a_view = A.view(torch.int32) if packed_path else A
        q_view = q.view(torch.int16) if packed_path else q
        kernel[40,](
            a_view,
            q_view,
            t,
            expert_offsets,
            problem_sizes,
            kg,
            e,
            triton.next_power_of_2(e),
            expert_offsets.stride(0),
            problem_sizes.stride(0),
            group_size,
            m * kg,
            ng,
            block,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _reorder[e, max(1, 40 // e)](
            t,
            s,
            expert_offsets,
            problem_sizes,
            kg,
            expert_offsets.stride(0),
            problem_sizes.stride(0),
            256,
            num_warps=4,
        )
    elif m * kg and e:
        packed = group_size == 128 and m * kg >= 8192
        kernel = _expert_paired if packed else _expert
        a_view = A.view(torch.int32) if packed else A
        q_view = q.view(torch.int16) if packed else q
        kernel[e, max(1, 40 // e)](
            a_view,
            q_view,
            s,
            expert_offsets,
            problem_sizes,
            kg,
            group_size,
            expert_offsets.stride(0),
            problem_sizes.stride(0),
            (64 if packed else ng),
            block,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return (q.view(torch.float8_e4m3fn), s)


@triton.jit
def _quant_range_paired(
    A,
    Q,
    T,
    table,
    start,
    count,
    G: tl.constexpr,
    TOTAL: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
):
    for tile in range(
        tl.program_id(0), tl.cdiv(count, NG), tl.num_programs(0)
    ):
        ids = tile * NG + tl.arange(0, NG)
        cols = tl.arange(0, B // 2)
        offsets = (start + ids)[:, None] * (G // 2) + cols[None, :]
        mask = (ids[:, None] < count) & (cols[None, :] < G // 2)
        words = tl.load(A + offsets, mask, 0)
        lo = words & 32767
        hi = words >> 16 & 32767
        amax = tl.maximum(
            tl.max(tl.maximum(lo, hi).to(tl.float32), 1).to(tl.int32) << 16,
            953267991,
        ).to(tl.float32, bitcast=True)
        inverse = _reference_inverse(amax, table)
        xlo = (lo << 16).to(tl.float32, bitcast=True)
        code_lo = _encode_positive_word(
            xlo * inverse[:, None], words >> 8 & 128
        )
        xhi = (hi << 16).to(tl.float32, bitcast=True)
        code_hi = _encode_positive_word(
            xhi * inverse[:, None], words >> 24 & 128
        )
        packed = (code_lo | code_hi << 8).to(tl.int16)
        tl.store(Q + offsets, packed, mask)
        tl.store(T + start + ids, tl.div_rn(amax, 448.0), ids < count)


@triton.jit
def _quant_paired(
    A,
    Q,
    T,
    OFF,
    PS,
    KG: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    G: tl.constexpr,
    TOTAL: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
):
    if TOTAL >= 98304:
        table = _computed_table()
    else:
        table = _mantissa_table()
    experts = tl.arange(0, BE)
    expert_starts = tl.load(OFF + experts * OS, experts < E, 0)
    counts = tl.load(PS + experts * PS0, experts < E, 0)
    ends = tl.load(OFF + (experts + 1) * OS, experts + 1 < E, TOTAL // KG)
    contiguous = (
        tl.sum(
            ((experts < E) & (expert_starts + counts != ends)).to(tl.int32), 0
        )
        == 0
    )
    contiguous = contiguous & (tl.load(OFF) == 0)
    if contiguous:
        for tile in range(
            tl.program_id(0), tl.cdiv(TOTAL, NG), tl.num_programs(0)
        ):
            ids = tile * NG + tl.arange(0, NG)
            cols = tl.arange(0, B // 2)
            offsets = ids[:, None] * (G // 2) + cols[None, :]
            mask = (ids[:, None] < TOTAL) & (cols[None, :] < G // 2)
            words = tl.load(A + offsets, mask, 0)
            lo = words & 32767
            hi = words >> 16 & 32767
            amax = tl.maximum(
                tl.max(tl.maximum(lo, hi).to(tl.float32), 1).to(tl.int32)
                << 16,
                953267991,
            ).to(tl.float32, bitcast=True)
            inverse = _reference_inverse(amax, table)
            xlo = (lo << 16).to(tl.float32, bitcast=True)
            code_lo = _encode_positive_word(
                xlo * inverse[:, None], words >> 8 & 128
            )
            xhi = (hi << 16).to(tl.float32, bitcast=True)
            code_hi = _encode_positive_word(
                xhi * inverse[:, None], words >> 24 & 128
            )
            packed = (code_lo | code_hi << 8).to(tl.int16)
            tl.store(Q + offsets, packed, mask)
            tl.store(T + ids, tl.div_rn(amax, 448.0), ids < TOTAL)
    else:
        for e in range(E):
            start = tl.load(OFF + e * OS) * KG
            count = tl.load(PS + e * PS0) * KG
            _quant_range_paired(A, Q, T, table, start, count, G, TOTAL, NG, B)


@triton.jit
def _encode_positive(v, sign):
    raw = v.to(tl.int32, bitcast=True)
    exponent = tl.maximum(raw >> 23, 121)
    bias = exponent * 8388616 + 167771192
    rounded = (v + bias.to(tl.float32, bitcast=True)).to(
        tl.int32, bitcast=True
    )
    code = rounded & 255 | sign
    return tl.where(v != v, 127, code).to(tl.uint8)


@triton.jit
def _encode_positive_word(v, sign):
    raw = v.to(tl.int32, bitcast=True)
    exponent = tl.maximum(raw >> 23, 121)
    bias = exponent * 8388616 + 167771192
    rounded = (v + bias.to(tl.float32, bitcast=True)).to(
        tl.int32, bitcast=True
    )
    code = rounded & 255 | sign
    return tl.where(v != v, 127, code).to(tl.int32)


@triton.jit
def _expert_paired(
    A,
    Q,
    S,
    OFF,
    PS,
    KG: tl.constexpr,
    G: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
):
    e = tl.program_id(0)
    start = tl.load(OFF + e * OS)
    count = tl.load(PS + e * PS0)
    table = _computed_table()
    for tile in range(
        tl.program_id(1), tl.cdiv(count * KG, NG), tl.num_programs(1)
    ):
        ids = tile * NG + tl.arange(0, NG)
        cols = tl.arange(0, B // 2)
        offsets = (start * KG + ids[:, None]) * (G // 2) + cols[None, :]
        mask = (ids[:, None] < count * KG) & (cols[None, :] < G // 2)
        words = tl.load(A + offsets, mask, 0)
        lo = words & 32767
        hi = words >> 16 & 32767
        amax = tl.maximum(
            tl.max(tl.maximum(lo, hi).to(tl.float32), 1).to(tl.int32) << 16,
            953267991,
        ).to(tl.float32, bitcast=True)
        inverse = _reference_inverse(amax, table)
        xlo = (lo << 16).to(tl.float32, bitcast=True)
        code_lo = _encode_positive_word(
            xlo * inverse[:, None], words >> 8 & 128
        )
        xhi = (hi << 16).to(tl.float32, bitcast=True)
        code_hi = _encode_positive_word(
            xhi * inverse[:, None], words >> 24 & 128
        )
        tl.store(Q + offsets, (code_lo | code_hi << 8).to(tl.int16), mask)
        dst = start * KG + ids % KG * count + ids // KG
        tl.store(S + dst, tl.div_rn(amax, 448.0), ids < count * KG)


@triton.jit
def _pair_rows(
    A,
    Q,
    S,
    OFF,
    PS,
    KG: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    R: tl.constexpr,
    COMPACT: tl.constexpr,
):
    expert = tl.program_id(0)
    start = tl.load(OFF + expert * OS)
    count = tl.load(PS + expert * PS0)
    table = _computed_table()
    lane = tl.arange(0, R)
    cols = tl.arange(0, 64)
    for tile in range(
        tl.program_id(1), tl.cdiv(count, R) * KG, tl.num_programs(1)
    ):
        rows = tile // KG * R + lane
        group = tile % KG
        valid = rows < count
        offsets = ((start + rows[:, None]) * KG + group) * 64 + cols[None, :]
        words = tl.load(A + offsets, valid[:, None], 0)
        lo = words & 32767
        hi = words >> 16 & 32767
        amax = tl.maximum(
            tl.max(tl.maximum(lo, hi).to(tl.float32), 1).to(tl.int32) << 16,
            953267991,
        ).to(tl.float32, bitcast=True)
        inverse = _reference_inverse(amax, table)
        xlo = (lo << 16).to(tl.float32, bitcast=True)
        code_lo = _encode_row_word(
            xlo * inverse[:, None], words >> 8 & 128, KG
        )
        if COMPACT:
            code_lo = code_lo.to(tl.int16)
        xhi = (hi << 16).to(tl.float32, bitcast=True)
        code_hi = _encode_row_word(
            xhi * inverse[:, None], words >> 24 & 128, KG
        )
        if COMPACT:
            code_hi = code_hi.to(tl.int16)
        tl.store(
            Q + offsets, (code_lo | code_hi << 8).to(tl.int16), valid[:, None]
        )
        tl.store(
            S + start * KG + group * count + rows,
            tl.div_rn(amax, 448.0),
            valid,
        )


@triton.jit
def _row_block(
    A, Q, S, table, start, count, tile, KG: tl.constexpr, R: tl.constexpr
):
    lane = tl.arange(0, R)
    cols = tl.arange(0, 64)
    rows = tile // KG * R + lane
    group = tile % KG
    valid = rows < count
    offsets = ((start + rows[:, None]) * KG + group) * 64 + cols[None, :]
    words = tl.load(A + offsets, valid[:, None], 0)
    lo = words & 32767
    hi = words >> 16 & 32767
    amax = tl.maximum(
        tl.max(tl.maximum(lo, hi).to(tl.float32), 1).to(tl.int32) << 16,
        953267991,
    ).to(tl.float32, bitcast=True)
    inverse = _reference_inverse(amax, table)
    xlo = (lo << 16).to(tl.float32, bitcast=True)
    code_lo = _encode_positive_word(xlo * inverse[:, None], words >> 8 & 128)
    xhi = (hi << 16).to(tl.float32, bitcast=True)
    code_hi = _encode_positive_word(xhi * inverse[:, None], words >> 24 & 128)
    tl.store(
        Q + offsets, (code_lo | code_hi << 8).to(tl.int16), valid[:, None]
    )
    tl.store(
        S + start * KG + group * count + rows, tl.div_rn(amax, 448.0), valid
    )


@triton.jit
def _pair_uniform(
    A,
    Q,
    S,
    OFF,
    PS,
    E: tl.constexpr,
    BE: tl.constexpr,
    KG: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    R: tl.constexpr,
):
    first_count = tl.load(PS)
    experts = tl.arange(0, BE)
    sizes = tl.load(PS + experts * PS0, experts < E, 0)
    uniform = (
        tl.sum(((experts < E) & (sizes != first_count)).to(tl.int32), 0) == 0
    )
    table = _computed_table()
    if uniform:
        chunks = tl.cdiv(first_count, R)
        if (chunks > 0) & (chunks & chunks - 1 == 0):
            shift = (
                chunks.to(tl.float32).to(tl.int32, bitcast=True) >> 23
            ) - 127
            for flat in range(
                tl.program_id(0), E * chunks * KG, tl.num_programs(0)
            ):
                local = flat // E
                expert = flat % E
                group = local >> shift
                row_chunk = local & (chunks - 1)
                start = tl.load(OFF + expert * OS)
                _row_coords(
                    A, Q, S, table, start, first_count, row_chunk, group, KG, R
                )
        else:
            for flat in range(
                tl.program_id(0), E * chunks * KG, tl.num_programs(0)
            ):
                expert = flat // (tl.maximum(chunks, 1) * KG)
                tile = flat % (tl.maximum(chunks, 1) * KG)
                start = tl.load(OFF + expert * OS)
                _row_block(A, Q, S, table, start, first_count, tile, KG, R)
    else:
        for expert in range(tl.program_id(0), E, tl.num_programs(0)):
            start = tl.load(OFF + expert * OS)
            count = tl.load(PS + expert * PS0)
            for tile in range(tl.cdiv(count, R) * KG):
                _row_block(A, Q, S, table, start, count, tile, KG, R)


@triton.jit
def _row_coords(
    A,
    Q,
    S,
    table,
    start,
    count,
    row_chunk,
    group,
    KG: tl.constexpr,
    R: tl.constexpr,
):
    lane = tl.arange(0, R)
    cols = tl.arange(0, 64)
    rows = row_chunk * R + lane
    valid = rows < count
    offsets = ((start + rows[:, None]) * KG + group) * 64 + cols[None, :]
    words = tl.load(A + offsets, valid[:, None], 0)
    lo = words & 32767
    hi = words >> 16 & 32767
    amax = tl.maximum(
        tl.max(tl.maximum(lo, hi).to(tl.float32), 1).to(tl.int32) << 16,
        953267991,
    ).to(tl.float32, bitcast=True)
    inverse = _reference_inverse(amax, table)
    xlo = (lo << 16).to(tl.float32, bitcast=True)
    code_lo = _encode_positive_word(xlo * inverse[:, None], words >> 8 & 128)
    xhi = (hi << 16).to(tl.float32, bitcast=True)
    code_hi = _encode_positive_word(xhi * inverse[:, None], words >> 24 & 128)
    tl.store(
        Q + offsets, (code_lo | code_hi << 8).to(tl.int16), valid[:, None]
    )
    tl.store(
        S + start * KG + group * count + rows, tl.div_rn(amax, 448.0), valid
    )


@triton.jit
def _large_address(
    A,
    Q,
    S,
    OFF,
    PS,
    E: tl.constexpr,
    KG: tl.constexpr,
    G: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
):
    table = _computed_table()
    lane = tl.arange(0, 128)
    for expert in range(tl.program_id(1).to(tl.int64), E, tl.num_programs(1)):
        start = tl.load(OFF + expert * OS).to(tl.int64)
        count = tl.load(PS + expert * PS0).to(tl.int64)
        for tile in range(
            tl.program_id(0).to(tl.int64),
            tl.cdiv(count * KG, 4),
            tl.num_programs(0),
        ):
            group = tile * 4 + tl.arange(0, 4)
            valid = group < count * KG
            base = (start * KG + group) * G
            maximum = tl.full((4, 128), 0, tl.float32)
            for chunk in range(tl.cdiv(G, 128)):
                col = chunk.to(tl.int64) * 128 + lane
                x = tl.load(
                    A + base[:, None] + col[None, :],
                    valid[:, None] & (col[None, :] < G),
                    0,
                ).to(tl.float32)
                maximum = tl.maximum(
                    maximum,
                    (x.to(tl.int32, bitcast=True) & 2147483647).to(tl.float32),
                )
            amax = tl.maximum(tl.max(maximum, 1).to(tl.int32), 953267991).to(
                tl.float32, bitcast=True
            )
            inverse = _reference_inverse(amax, table)
            dst = start * KG + (group % KG) * count + group // KG
            tl.store(S + dst, tl.div_rn(amax, 448.0), valid)
            for chunk in range(tl.cdiv(G, 128)):
                col = chunk.to(tl.int64) * 128 + lane
                x = tl.load(
                    A + base[:, None] + col[None, :],
                    valid[:, None] & (col[None, :] < G),
                    0,
                ).to(tl.float32)
                tl.store(
                    Q + base[:, None] + col[None, :],
                    _encode(x * inverse[:, None]),
                    valid[:, None] & (col[None, :] < G),
                )


def _large_entry(A, expert_offsets, problem_sizes, group_size):
    m, k = A.shape
    kg = k // group_size
    e = problem_sizes.shape[0]
    q = torch.empty((m, k), dtype=torch.uint8, device=A.device)
    s = torch.empty((m, kg), dtype=torch.float32, device=A.device)
    if m * kg and e:
        _large_address[(32, min(e, 40))](
            A,
            q,
            s,
            expert_offsets,
            problem_sizes,
            e,
            kg,
            group_size,
            expert_offsets.stride(0),
            problem_sizes.stride(0),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q.view(torch.float8_e4m3fn), s


@triton.jit
def _encode_row_word(v, sign, KG: tl.constexpr):
    if KG > 24:
        return _encode_positive_word(v, sign)
    raw = v.to(tl.int32, bitcast=True)
    exponent = tl.maximum(raw >> 23, 121)
    bias = exponent * 8388616 + 167771192
    rounded = (v + (bias | sign).to(tl.float32, bitcast=True)).to(
        tl.int32, bitcast=True
    )
    code = rounded & 255
    return tl.where(v != v, 127, code).to(tl.int32)


__all__ = ["per_token_group_quant_fp8_moe_mn_major"]
