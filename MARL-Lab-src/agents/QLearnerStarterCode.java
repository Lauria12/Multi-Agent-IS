package agents;

import java.util.Random;
import java.util.Arrays;

public class QLearnerStarterCode implements Agent {

	private double[] actionProbabilities;
	private double learningRate, minProb;
	private Random random;

	public QLearnerStarterCode(int numberOfActions) {
		actionProbabilities = new double[numberOfActions];
		Arrays.fill(actionProbabilities, 1.0 / numberOfActions); // Initialize uniform probabilities

		learningRate = 0.05; // Increased learning rate for better adaptation
		minProb = 0.05; // Reduced minimum probability for better exploitation

		random = new Random();
	}

	// Returns the probability of selecting a given action
	@Override
	public double actionProb(int index) {
		return actionProbabilities[index];
	}

	// Select action based on probabilities
	@Override
	public int selectAction() {
		double r = random.nextDouble();
		double cumulative = 0.0;
		for (int i = 0; i < actionProbabilities.length; i++) {
			cumulative += actionProbabilities[i];
			if (r <= cumulative) {
				return i;
			}
		}
		return actionProbabilities.length - 1; // Fallback in case of precision issues
	}

	// Learning Automata update rule with more adaptive changes
	@Override
	public void update(int ownAction, int opponentAction, double reward) {
		// More adaptive probability updates
		if (reward > 0) {
			actionProbabilities[ownAction] += learningRate * (1.0 - actionProbabilities[ownAction]);
		} else {
			actionProbabilities[ownAction] -= learningRate * (actionProbabilities[ownAction] - minProb);
		}

		// Normalize probabilities to ensure they sum to 1
		double sum = Arrays.stream(actionProbabilities).sum();
		for (int i = 0; i < actionProbabilities.length; i++) {
			actionProbabilities[i] /= sum;
		}
	}

	// Returns action probability (as a replacement for Q-values)
	@Override
	public double getQ(int index) {
		return actionProbabilities[index]; // Returning probability instead of Q-value
	}
}
