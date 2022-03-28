
/*
 * Testiranje:
 *
 * tj.exe
 */

import java.util.*;

public class Druga {

    public static void main(String[] args) {
        // koda za ro"cno testiranje (po "zelji)
    }

    public static int najCas(int[][] t, int krog) {
        int najCas = t[0][krog];
		for(int i = 0; i < t.length; i++){
			if(t[i][krog] < najCas){
				najCas = t[i][krog];
			}
		}
        return najCas;
    }

    public static int[][] kumulativa(int[][] t) {
        int[][] rezultat = new int[t.length][t[0].length];
		for(int i = 0; i < t.length; i++){
			rezultat[i][0] = t[i][0];
			for(int j = 1; j < t[i].length; j++){
				rezultat[i][j] = rezultat[i][j-1] + t[i][j];
			}
		}
        return rezultat;
    }
}
