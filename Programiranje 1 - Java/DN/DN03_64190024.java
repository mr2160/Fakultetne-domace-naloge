import java.util.Scanner;

public class DN03_64190024 {
	public static void main(String[] args){
		
		Scanner sc = new Scanner(System.in);
		int visina = -(sc.nextInt()-1);
		int sirina = sc.nextInt()-1;
		int aV = sc.nextInt();
		int bV = sc.nextInt();
		int cV = sc.nextInt();
		
		int aJ = sc.nextInt();
		int bJ = sc.nextInt();
		int cJ = sc.nextInt();
		int casP = sc.nextInt();
		
		int cas = 0;
		
		int trPolozajX = 0;
		int trPolozajY = 0;
		
		int sPolozajX = sirina;
		int sPolozajY = visina;
		
		while (trPolozajX != sPolozajX || trPolozajY != sPolozajY) {
			//System.out.printf("Lokacija: %d, %d%n", trPolozaj[0], trPolozaj[1]);
			//System.out.printf("Čas: %d%n", cas);
			if (trPolozajX == sirina && zelenaJ(cas, aJ, bJ, cJ)) {
				trPolozajY -= 1;
				cas += casP;
				//System.out.println("Preckal dol");
			} else if (trPolozajY == visina && zelenaV(cas, aV, bV, cV)) {
				trPolozajX += 1;
				//System.out.println("Preckal desno");
				cas += casP;
			} else if (zelenaV(cas, aV, bV, cV) && trPolozajX != sirina) {
				trPolozajX += 1;
				//System.out.println("Preckal desno");
				cas += casP;
			} else if (zelenaJ(cas, aJ, bJ, cJ) && (zelenaV(cas,aV,bV,cV)!= true) && trPolozajY != visina) {
				trPolozajY -= 1;
				//System.out.println("Preckal dol");
				cas += casP;
			} else {
				cas +=1;
				//System.out.println("Čakal");
			}
		}
		
		System.out.println(cas);
		
		
	}

	public static boolean zelenaV(int cas, int aV, int bV, int cV) {
		cas = cas - aV;
		int perioda = bV + cV;
		int casPerioda = cas%perioda;
		if (cas < 0) {
			return false;
		} else if(casPerioda < bV) {
			//System.out.printf("Vzhodne prižgane. Čas: %d%n", cas+aV);
			return true;
		} else{
			return false;
		}
			
	}
		
	public static boolean zelenaJ(int cas, int aJ, int bJ, int cJ) {
		cas = cas - aJ;
		int perioda = bJ + cJ;
		int casPerioda = cas%perioda;
		if (cas < 0) {
			return false;
		} else if(casPerioda < bJ) {
			//System.out.printf("Južne prižgane. Čas: %d%n", cas+aJ);
			return true;
		} else{
			return false;
		}
			
	}
}	
