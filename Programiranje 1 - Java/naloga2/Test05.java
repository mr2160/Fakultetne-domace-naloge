
public class Test05 {

    public static void main(String[] args) {
        int[][] t = {
            { 29, 70, 23, 37, 23, 60, 43, 87, 20, 28, 72, 39, 41 },
            { 30, 26, 82, 68, 44, 21, 62, 28, 35, 33, 11, 95, 32 },
            { 44, 40, 31, 12, 92, 67, 49, 74, 23, 96, 26, 20, 75 },
            { 92,  9, 70, 80, 59, 15, 77, 26, 49,  3, 19, 34, 63 },
            { 92, 70, 92, 56, 16, 59, 13, 93,  6, 17, 60, 71, 88 },
            { 23, 42, 17, 71, 61, 81, 15, 43,  4, 95, 11, 61, 55 },
        };

        for (int krog = 0;  krog < 13;  krog++) {
            System.out.println(Druga.najCas(t, krog));
        }
    }
}
