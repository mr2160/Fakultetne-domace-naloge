
public class Test11 {

    public static void main(String[] args) {
        int[][] t = {
            { 4716, 2380, 9682, 7149, 5102, 4014, 1319 },
            { 2890, 4052, 6619, 3108, 7060, 2099, 7923 },
            { 4221, 8493, 7733, 4160, 3054, 8893, 9872 },
            {  553, 7466,  215,  200, 2060, 6900, 5860 },
            { 1317,  295, 4084, 7106, 6199, 1582, 4924 },
            { 4144, 5424,  417,  598, 9123, 8924, 8076 },
            { 1117, 2095, 3570, 4161, 9821, 1469, 8795 },
            { 4982, 7129, 3278, 4822, 8037, 2213, 7174 },
            { 6461, 2030, 9048, 8141, 4883, 1920, 3981 },
            { 2636, 6740, 6606, 4537, 7967, 1581, 7171 },
        };

        for (int krog = 0;  krog < 7;  krog++) {
            System.out.println(Druga.najCas(t, krog));
        }
    }
}