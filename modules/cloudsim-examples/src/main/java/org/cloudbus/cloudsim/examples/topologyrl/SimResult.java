package org.cloudbus.cloudsim.examples.topologyrl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SimResult {
    public final String schedulerName;
    public final double lambdaPerHour;

    public int totalArrivals = 0;
    public int accepted = 0;
    public int rejected = 0;
    public int slaViolations = 0;

    public double totalCostCny = 0.0;

    private final List<Double> e2eDelays = new ArrayList<>();
    private final List<Double> controlDelaysMs = new ArrayList<>();

    public SimResult(String schedulerName, double lambdaPerHour) {
        this.schedulerName = schedulerName;
        this.lambdaPerHour = lambdaPerHour;
    }

    public void addDelay(double delayMs) {
        e2eDelays.add(delayMs);
    }

    public double avgE2eDelayMs() {
        if (e2eDelays.isEmpty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (double d : e2eDelays) {
            sum += d;
        }
        return sum / e2eDelays.size();
    }

    public double p95E2eDelayMs() {
        if (e2eDelays.isEmpty()) {
            return 0.0;
        }
        List<Double> sorted = new ArrayList<>(e2eDelays);
        Collections.sort(sorted);
        int idx = (int) Math.floor(0.95 * (sorted.size() - 1));
        return sorted.get(idx);
    }

    public void addControlDelayMs(double delayMs) {
        controlDelaysMs.add(delayMs);
    }

    public double avgControlDelayMs() {
        if (controlDelaysMs.isEmpty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (double d : controlDelaysMs) {
            sum += d;
        }
        return sum / controlDelaysMs.size();
    }

    public double p95ControlDelayMs() {
        if (controlDelaysMs.isEmpty()) {
            return 0.0;
        }
        List<Double> sorted = new ArrayList<>(controlDelaysMs);
        Collections.sort(sorted);
        int idx = (int) Math.floor(0.95 * (sorted.size() - 1));
        return sorted.get(idx);
    }
}
