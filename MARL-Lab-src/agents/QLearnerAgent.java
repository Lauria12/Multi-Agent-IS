package agents;

import java.util.Random;

public class QLearnerAgent implements Agent {

    private double[] Q;
    private double alpha, gamma, tau, tauDecay;
    private Random random;

    public QLearnerAgent(int numberOfActions) {
        Q = new double[numberOfActions];
        for (int i = 0; i < numberOfActions; i++)
            Q[i] = -0.1 + Math.random() * 0.2; // Initialize Q-values randomly

        alpha = 0.001; // Learning rate
        gamma = 0.9; // Discount factor
        tau = 0.2; // Initial temperature for Boltzmann exploration
        tauDecay = 0.999; // Decay rate for temperature

        random = new Random();
    }

    // Returns the probability of selecting a given action using Boltzmann exploration
    public double actionProb(int index) {
        double sumExp = 0.0;
        for (double q : Q) {
            sumExp += Math.exp(q / tau);
        }
        return Math.exp(Q[index] / tau) / sumExp;
    }

    // Select action using Boltzmann exploration
    public int selectAction() {
        double[] probabilities = new double[Q.length];
        double sumExp = 0.0;

        for (double q : Q) {
            sumExp += Math.exp(q / tau);
        }

        for (int i = 0; i < Q.length; i++) {
            probabilities[i] = Math.exp(Q[i] / tau) / sumExp;
        }

        double r = random.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];
            if (r <= cumulative) {
                return i;
            }
        }
        return Q.length - 1; // Fallback in case of floating-point precision issues
    }

    // Q-learning update rule
    public void update(int ownAction, int opponentAction, double reward) {
        Q[ownAction] = Q[ownAction] + alpha * (reward - Q[ownAction]);
        tau *= tauDecay; // Decay temperature over time
    }

    // Returns Q-value for a given action
    public double getQ(int index) {
        return Q[index];
    }
}
