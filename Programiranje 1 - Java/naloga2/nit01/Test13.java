
public class Test13 {

    public static void main(String[] args) {
        int[][] t = {
            { 9178, 4886, 6844, 5745, 9403, 3573, 6800, 2651, 2030, 5779, 6298, 6473,  736,  100,  576, 2708, 2173, 4417, 4330, 5603, 1777, 5175, 8965 },
            { 4113, 3372, 1367,  584, 2706, 5904, 4624, 9669, 7435, 2964, 3220, 1390, 8612, 6016, 2136, 1485,  581, 2923, 7936, 9473, 1740, 5405, 1003 },
            { 4191, 3076, 4135, 4786, 7068, 3139,  506, 7256, 9654, 7456, 5815, 3498, 1782, 8801, 1161, 6189, 7423, 5532, 5849, 8181, 6528, 3307, 3506 },
            { 6558, 9132, 7930, 8367, 7815, 9973, 3500,  909, 8828, 2396, 1441, 2991, 8947, 7003, 4660, 3376, 5006, 2827, 5409, 9487, 8162, 9231, 2628 },
            { 3635,  268,  623, 9647, 4413, 8999, 6575, 4697,  129, 3312, 7577, 2474, 3808, 5377, 7483, 1916, 6982, 9830, 5781, 4420, 2119, 3598, 1694 },
            { 2408, 5164, 2371, 9605, 6288, 4255, 4742, 1473, 6009,  596, 9455, 2291, 5983, 1368, 7364, 2878,  554, 5339, 6819, 1608, 7183,  742, 5695 },
            { 1140, 3911, 6671, 2397, 3047, 9429, 2951, 8126, 8935, 5783, 1075, 7750, 7146, 6987, 2455, 9884, 7509, 1751, 2512, 4256,  330, 7785, 4576 },
            { 1972, 7104, 8346, 2489, 4638, 9491, 1027,  886, 6344, 9039, 7898, 7365, 6787, 2610, 2872, 3398, 2475, 8950, 6368, 1488,  379, 3810,  204 },
        };

        for (int krog = 0;  krog < 23;  krog++) {
            System.out.println(Druga.najCas(t, krog));
        }
    }
}