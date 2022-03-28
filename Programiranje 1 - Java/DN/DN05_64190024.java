import java.util.Scanner;
import java.util.Arrays;
import java.util.ArrayList;
public class DN05_64190024 {
	public static void main(String[] args){
		Scanner sc = new Scanner(System.in);
		
		int dolzina = 1;
		int[] trak = new int[dolzina];
		int polozajGlave = 0;
		
		String program = sc.next();
		int koraki = 0;
		for(int i = 0; i < program.length(); i++){
			int ukaz = program.charAt(i);
			if(koraki >= 10000){
				System.out.println(trak[polozajGlave]);
				break;
			}
			//System.out.printf("izvajam: %c pri: %d%n", (char) ukaz, polozajGlave);
			switch (ukaz){
				case '+':
					trak = plus(trak, polozajGlave);
				break;
				case '-':
					trak = minus(trak, polozajGlave);
				break;
				case '>':
					if(polozajGlave == dolzina-1){
						trak = povecajGor(trak, dolzina);
						dolzina = trak.length;
					}
					polozajGlave += 1;
				break;
				case'<':
					if(polozajGlave == 0){
						trak = povecajDol(trak, dolzina);
						dolzina = trak.length;
						polozajGlave = dolzina/2;
					}
					polozajGlave -= 1;
				break;
				case '.':
					System.out.println(trak[polozajGlave]);
				break;
				case ',':
					if(sc.hasNext()){
						trak[polozajGlave] = sc.nextInt();
					}
					else{
						trak[polozajGlave] = 0;
					}
				break;
				case '[':
					if(trak[polozajGlave] == 0) {
						i = poisciZaklepaj(program, i) - 1;
					}
				break;
				case ']':
					if(trak[polozajGlave] != 0) {
						i = poisciOklepaj(program, i) + 1;
					}
				break;
				case '*':
					System.out.println(Arrays.toString(trak));
				break;
			}
			//System.out.println(Arrays.toString(trak));
			koraki++;
		}
		
		
		//trak = povecajGor(trak, dolzina);
		//dolzina = trak.length;
		
	}
	
	public static int[] povecajGor(int[] trak, int dolzina) {
		int[] novT = new int[dolzina+4];
		for(int i = 0; i < dolzina; i++) {
			novT[i] = trak[i];
		}
		return novT;
	}
	
	public static int[] povecajDol(int[] trak, int dolzina) {
		int[] novT = new int[dolzina+4];
		for(int i = 0; i < dolzina; i++) {
			novT[i+4] = trak[i];
		}
		return novT;
	}
	
	public static int[] plus(int[] trak, int polozajGlave){
		if(trak[polozajGlave] == 255){
			trak[polozajGlave] = 0;
		}else{
			trak[polozajGlave] += 1;
		}
		return(trak);
	}
	
	public static int[] minus(int[] trak, int polozajGlave){
		if(trak[polozajGlave] == 0){
			trak[polozajGlave] = 255;
		}else{
			trak[polozajGlave] -= 1;
		}
		return(trak);
	}
	
	public static int poisciZaklepaj(String program, int i){
		int oklepaji = 1;
		int zaklepaji = 0;
		i++;
		while (oklepaji != zaklepaji) {
			//System.out.println("i:" + i);
			//System.out.println("ukaz:" + program.charAt(i));
			if (program.charAt(i) == '['){
				oklepaji += 1;
			} else if (program.charAt(i) == ']'){
				zaklepaji += 1;
			}
			i++;
		}
		return i;
	}

	public static int poisciOklepaj(String program, int i){
		int oklepaji = 0;
		int zaklepaji = 1;
		i--;
		while (oklepaji != zaklepaji) {
			//System.out.println("i:" + i);
			//System.out.println("ukaz:" + program.charAt(i));
			if (program.charAt(i) == '['){
				oklepaji += 1;
			} else if (program.charAt(i) == ']'){
				zaklepaji += 1;
			}
			i--;
		}
		return i;
	}
}	
