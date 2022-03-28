import java.util.Scanner;

public class DN01_64190024 {
	public static void main(String[] args){
		Scanner sc = new Scanner(System.in);
		int telo = sc.nextInt();
		switch (telo) {
			
			case 1: {
				int a = sc.nextInt();
				System.out.println(kocka(a));
				break;
			}
			
			case 2: {
				int a = sc.nextInt();
				int b = sc.nextInt();
				int c = sc.nextInt();
				System.out.println(kvader(a, b, c));
				break;
			}
			
			case 3: {
				int a = sc.nextInt();
				System.out.println(piramida(a));
				break;
			}
			
			case 4: {
				int a = sc.nextInt();
				int b = sc.nextInt();
				System.out.println(prizma(a, b));
				break;
			}
			case 5: {
				int a = sc.nextInt();
				int volumen = piramida(a) - piramida(a-2);
				System.out.println(volumen);
				break;
			}
			case 6: {
				int a = sc.nextInt();
				int b = sc.nextInt();
				int volumen = prizma(a, b) - prizma(a-2, b-2);
				System.out.println(volumen);
				break;
			}
			
		}
		
		
	} 
	
	public static int kocka(int stranica) {
		int volumen = stranica*stranica*stranica;
		return volumen;
	}
	
	public static int kvader(int v, int s, int d){
		int volumen = v*s*d;
		return volumen;
	}
	
	public static int piramida(int s){
		int volumen = 0;
		for (int i=s; i>0; i-=1){
			volumen += i*i;
		}
		return volumen;
	}
	
	public static int prizma(int s, int d){
		int volumen = 0;
		for (int i=s; i>0; i-=1){
			volumen += i*d;
		}
		return volumen;
	}
}