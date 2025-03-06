package agents;

import java.util.Random;
import java.util.Arrays;

public class QLearnerStarterCode implements Agent {

	private double[] actionProbabilities;
	private double learningRate, minProb, explorationRate;
	private Random random;
	private int iterationCount;

	public QLearnerStarterCode(int numberOfActions) {
		actionProbabilities = new double[numberOfActions];
		Arrays.fill(actionProbabilities, 1.0 / numberOfActions);

		learningRate = 0.05;
		minProb = 0.1;
		explorationRate = 0.5;
		random = new Random();
		iterationCount = 1;
	}

	@Override
	public double actionProb(int index) {
		return actionProbabilities[index];
	}

	@Override
	public int selectAction() {
		if (random.nextDouble() < explorationRate) {
			return random.nextInt(actionProbabilities.length);
		}

		double r = random.nextDouble();
		double cumulative = 0.0;
		for (int i = 0; i < actionProbabilities.length; i++) {
			cumulative += actionProbabilities[i];
			if (r <= cumulative) {
				return i;
			}
		}
		return actionProbabilities.length - 1;
	}

	@Override
	public void update(int ownAction, int opponentAction, double reward) {
		if (reward > 0) {
			actionProbabilities[ownAction] += learningRate * (1.0 - actionProbabilities[ownAction]);
		} else {
			actionProbabilities[ownAction] -= learningRate * (actionProbabilities[ownAction] - minProb);
		}

		double sum = Arrays.stream(actionProbabilities).sum();
		for (int i = 0; i < actionProbabilities.length; i++) {
			actionProbabilities[i] /= sum;
		}

		// Keep exploration rate higher to react better
		iterationCount++;
		explorationRate = Math.max(0.1, 0.5 / Math.sqrt(iterationCount));
	}

	@Override
	public double getQ(int index) {
		return actionProbabilities[index];
	}
}
