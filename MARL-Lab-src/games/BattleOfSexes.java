package games;

public class BattleOfSexes implements Game {

	@Override
	public double[] move(int first, int second) {
		if (first==0 && second==0) {
			double[] res = {1.0, 0.5};
			return res;
		} else if (first==1 && second==1){
			double[] res = {0.5, 1.0};
			return res;
		} else {
			double[] res = {0.0,0.0};
			return res;
		}
	}

}
