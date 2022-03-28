
/*
 * Testiranje:
 *
 * tj.exe
 */

import java.util.*;

public class Tretja {

    public static void main(String[] args) {
        // koda za ro"cno testiranje (po "zelji)
    }

    public static class Zaposleni {
        private String ip;
        private int placa;
        private Vodja nadrejeni;

        public Zaposleni(String ip, int placa, Vodja nadrejeni) {
            this.ip = ip;
            this.placa = placa;
            this.nadrejeni = nadrejeni;
        }

        @Override
        public String toString() {
            return this.ip;
        }
		
		public Zaposleni vrniNad(){
			return this.nadrejeni;
		}
		
		public int vrniPlaco(){
			return this.placa;
		}

        public int placaNadrejenega() {
            if(this.vrniNad() == null){
				return -1;
			}
			return this.vrniNad().vrniPlaco();
        }

        public static int steviloAnomalij(Zaposleni[] zaposleni) {
            int stevec = 0;
			for(Zaposleni delavec: zaposleni){
				if(delavec instanceof Delavec){
					for(Zaposleni vodja: zaposleni){
						if(vodja instanceof Vodja){
							if(delavec.vrniPlaco() > vodja.vrniPlaco()){
								stevec++;
							}
						}
					}
				}
			}
            return stevec;
        }
    }

    public static class Delavec extends Zaposleni {
        public Delavec(String ip, int placa, Vodja nadrejeni) {
            super(ip, placa, nadrejeni);
        }
    }

    public static class Vodja extends Zaposleni {
        public Vodja(String ip, int placa, Vodja nadrejeni) {
            super(ip, placa, nadrejeni);
        }

        public Vodja vrhovni() {
            if(this.vrniNad() == null){
				return this;
			}
			Vodja nad = (Vodja) this.vrniNad();
            return nad.vrhovni();
        }
    }
}
