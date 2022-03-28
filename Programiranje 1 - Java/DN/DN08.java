import java.util.Scanner;
import java.util.Arrays;
public class DN08 {
	public static void main (String[] args){
		Scanner sc = new Scanner(System.in);
	
		/*int visina = sc.nextInt();
		int sirina = sc.nextInt();
		
		int[] rezultat = new int[sirina];
		
		for(int i = 0; i < sirina; i++){
			rezultat[i] = sc.nextInt();
		}
		
		
		for(int i=1; i < visina; i++){
			for(int j = 0; j < sirina; j++){
				int t = sc.nextInt();
				if(t > rezultat[j]){
					rezultat[j] = t;
				}
			}
		}
		System.out.println(Arrays.toString(rezultat));*/
		System.out.print(test());
	}
	
	public static String test(){
		return String.format("%s %d %d%n", "lol", 123, 'k');
	}
}