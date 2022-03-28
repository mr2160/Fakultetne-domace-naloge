
/*
 * Testiranje:
 *
 * tj.exe
 */

import java.util.*;

public class Cetrta {

    public static void main(String[] args) {
        // koda za ro"cno testiranje (po "zelji)
    }

    public static class Celica implements Comparable<Celica> {
        private int vrstica;
        private int stolpec;

        public Celica(int vrstica, int stolpec) {
            this.vrstica = vrstica;
            this.stolpec = stolpec;
        }
		
		public int vrniVrst(){
			return this.vrstica;
		}
		
		public int vrniStolp(){
			return this.stolpec;
		}

        @Override
        public String toString() {
            return String.format("(%d, %d)", this.vrstica, this.stolpec);
        }
		
		@Override
		public int compareTo(Celica a){
			if(this.vrniVrst() == a.vrniVrst()){
				return this.vrniStolp() - a.vrniStolp();
			}
			return this.vrniVrst() - a.vrniVrst();
		}
    }

    public static class Ovojnik {
        private boolean[][] tabela;

        public Ovojnik(boolean[][] tabela) {
            this.tabela = tabela;
        }
		

        public NavigableSet<Celica> enice() {
            Celica sredinska = new Celica((this.tabela.length/2)+1, (this.tabela[0].length/2)+1);
			NavigableSet<Celica> rezultat = new TreeSet<>((a, b) -> (manRazdalja(a, sredinska) - manRazdalja(b, sredinska)));
            
			for(int i = 0; i < this.tabela.length; i++){
				for(int j = 0; j < this.tabela[0].length; j++){
					if(this.tabela[i][j]){
						rezultat.add(new Celica(i, j));
					}
				}
			}
			return rezultat;
        }

        public static int abs(int n){
			if(n<0){
				return -n;
			}
			return n;
		}
		
		public static int manRazdalja(Celica a, Celica b){
			int aV = a.vrniVrst();
			int aS = a.vrniStolp();
			int bV = b.vrniVrst();
			int bS = b.vrniStolp();
			return abs(aV - bV) + abs(aS - bS);
		}
    }
}
