package org.cloudbus.cloudsim.examples.topologyrl;

public class SimConfig {
    public int numNodes = 50;
    public int numPotentialUsers = 100;
    public int avgCandidateNodes = 5;

    public int slotMinutes = 1;
    public int simDays = 3;

    public double bandwidthMbps = 100.0;
    public double slaMs = 50.0;

    public double nodeCostCnyPerHour = 5.35;
    public double lambdaPerHour = 8.0;

    public double tierLRatio = 0.6;
    public double tierHRatio = 0.4;

    public double sessionHoursMin = 1.0;
    public double sessionHoursMax = 3.0;

    public double gpuTflops = 31.2;

    public int decisionTimeoutMs = 200;
    public String pythonPolicyUrl = "http://127.0.0.1:8000/act";

    public LocalRegressionParams localRegression = new LocalRegressionParams();

    public int totalSlots() {
        return simDays * 24 * 60 / slotMinutes;
    }

    public double nodeCostPerMinute() {
        return nodeCostCnyPerHour / 60.0;
    }

    public static class LocalRegressionParams {
        public double a0 = 0.0;
        public double a1 = 20.0;
        public double a2 = 1.5;
        public double a3 = 8.0;
        public double a4 = 1.0;

        public double alpha = 1.0;
        public double beta = 0.8;
        public double gamma = 0.2;
    }
}
