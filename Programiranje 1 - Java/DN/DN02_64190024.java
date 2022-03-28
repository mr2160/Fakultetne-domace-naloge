import java.util.Scanner;

public class DN02_64190024 {
	public static void main(String[] args){
		
		Scanner sc = new Scanner(System.in);
		int ukaz = sc.nextInt();
		int a = sc.nextInt();
		int b = sc.nextInt();
		int k = sc.nextInt();
		
		switch (ukaz) {
			
			case 1: {
				System.out.println(deljivost(a, b, k));
				break;
			}
			
			case 2: {
				int vsota = 0;
				for (int i = a; i<= b; i++){
					if (vsebujeStevko(i, k)){
						vsota++;
					} 
				}
				System.out.println(vsota);
				break;
			}
			
			case 3: {
				int vsota = 0;
				for (int i = a; i<= b; i++){
					if (deljivostStevk(i, k)){
						vsota++;
					} 
				}
				System.out.println(vsota);
				break;
			}
			
			case 4: {
				int vsota = 0;
				for (int i = a; i<= b; i++){
					if (dolzinaNiza(i) >= k){
						vsota++;
					} 
				}
				System.out.println(vsota);
				break;
			}
			case 5: {
				int vsota = 0;
				for (int i = a; i<= b; i++){
					if (dolzinaZap(i) >= k){
						vsota++;
					} 
				}
				System.out.println(vsota);
				break;
			}			
		}		
	} 
	
	public static int deljivost(int a, int b, int k) {
		int vsota = 0;
		for (int i = a; i <= b; i++){
			if (i%k == 0){
				vsota++;
			}
		}
		return vsota;
	}
	
	public static boolean vsebujeStevko(int n, int k) {
		while (n > 0) {
			if ((n%10) == k){
				return true;
			} else {
				n = n/10;
			}
		}
	return false;	
	}

	public static boolean deljivostStevk(int n, int k) {
		while (n > 0) {
			if ((n%10)%k != 0){
				return false;
			} else {
				n = n/10;
			}
		}
	return true;	
	}
	
	public static int dolzinaNiza(int n) {
		int prejsnjaSt = n%10;
		n = n/10;
		int dolzina = 1;
		int najdaljsa = 1;
		
		while (n > 0){
			if ((n%10)==prejsnjaSt){
				dolzina++;
				n = n/10;
			} else {
				prejsnjaSt = n%10;
				n = n/10;
				dolzina = 1;
			}
			
			if (dolzina > najdaljsa){
				najdaljsa = dolzina;
			}
			
		}
		return najdaljsa;
	}
	
	public static int dolzinaZap(int n) {
		int prejsnjaSt = n%10;
		n = n/10;
		int dolzinaGor = 1;
		int dolzinaDol = 1;
		int najdaljsa = 1;
	
		while ( n > 0){
			if ((n%10)==prejsnjaSt-1){
				dolzinaDol++;
				dolzinaGor = 1;
				prejsnjaSt = n%10;
				n=n/10;
			} else if ((n%10)==prejsnjaSt+1){
				dolzinaGor++;
				dolzinaDol = 1;
				prejsnjaSt = n%10;
				n=n/10;
			} else {
				prejsnjaSt = n%10;
				n = n/10;
				dolzinaGor = 1;
				dolzinaDol = 1;
			}
			
			if(dolzinaDol > najdaljsa){
				najdaljsa = dolzinaDol;
			} else if (dolzinaGor > najdaljsa) {
				najdaljsa = dolzinaGor;
			}
		}
		return najdaljsa;
	}
}

	
	