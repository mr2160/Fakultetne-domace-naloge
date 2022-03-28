import java.util.Scanner;

public class Piramida {
	public static void main(String[] args){
		Scanner sc = new Scanner(System.in);
		int stVrstic = sc.nextInt();
	
		for(int i = 1; i <= stVrstic; i++){
			presledki(stVrstic, i);
			System.out.print(stevilke(i));
			System.out.println();
		}
	}
	
	public static String stevilke(int vrstica){
		String odgovor = "";
		for(int i = 1; i <= (2*vrstica)-1; i++){
			int stevilka = i;
			odgovor = odgovor + stevilka;
			stevilka += 1;
		}
		return odgovor;
	}
	
	public static void presledki(int stVrstic, int vrstica){
		for(int i = 0; i < (stVrstic - vrstica); i++){
			System.out.print("_");
		}
	}
}