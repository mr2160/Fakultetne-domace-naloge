import java.util.Scanner;
import java.util.Arrays;
public class DN04_64190024 {
	public static void main(String[] args){
		Scanner sc = new Scanner(System.in);
		
		int izvorniS = sc.nextInt();
		int outputS = sc.nextInt();
		int dolzina = sc.nextInt();
		
		char[] stevke = new char[dolzina];

		for(int i = 0; i < dolzina; i++){
			stevke[i] = sc.next().charAt(0);
		}
		
		int v10 = v10(stevke, izvorniS);
		System.out.print(iz10( v10, outputS, dolzinaAlg(v10, outputS)));
	}
	
	public static int v10(char[] stevke, int izvorniS) {
		int pretvorjena = 0;
		int stevka;
		for(int i = 0; i < stevke.length; i++){
			stevka = charVint(stevke[i]);
			pretvorjena += stevka * (potenca(izvorniS, (stevke.length-i-1)));
			//System.out.printf("trenutni izračun: %d, %d*%d**%d%n", pretvorjena, stevka, izvorniS, stevke.length-i-1);
		}
		return pretvorjena;
	}
	
	public static char[] iz10(int stevilo, int outputS, int dolzinaAlg) {
		char[] pretvorjena = new char[dolzinaAlg];
		int k = stevilo;
		for(int i = dolzinaAlg-1; i >= 0; i--){
			pretvorjena[i] = intVchar(k%outputS);
			k = k/outputS;
		}
		return pretvorjena;
	}
	
	public static int dolzinaAlg (int stevilka, int baza){
		int i = 0;
		int k = stevilka;
		while (k > 0){
			k = k/baza;
			i++;
		}
		return i;
	}

	
	public static int potenca(int osnova, int exponent){
		if (exponent == 0) {
			return 1;
		}
		int potenca = osnova;
		for(int i = 1; i < exponent; i++){
			potenca *= osnova;
		}
		return potenca;
	}
	
	public static int charVint(char stevka){
		int pretvorjena;
		if (stevka < 65) {
			pretvorjena = (int) stevka - 48;
		} else {
			pretvorjena = (int) stevka - 55;
		}
		
		return pretvorjena;
	}
	
	public static char intVchar(int stevka){
		char pretvorjena;
		if (stevka < 10) {
			pretvorjena = (char) (stevka + 48);
		} else {
			pretvorjena = (char) (stevka + 55);
		}
		
		return pretvorjena;
	}
}	
