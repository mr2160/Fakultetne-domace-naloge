
/*
 * Testiranje:
 *
 * tj.exe Prva.java . .
 */

import java.util.*;

public class Prva {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
		int a = sc.nextInt();
		int b = sc.nextInt();
		int k = sc.nextInt();
		
		int velikostT = a;
		int trenutniKup = 0;
		while(k > 0){
			trenutniKup += 1;
			k -= velikostT;
			velikostT += b;
		}
	
	System.out.println(trenutniKup);
    }
	
	
	
}
