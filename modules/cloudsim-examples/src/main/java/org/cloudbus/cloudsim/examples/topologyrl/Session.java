package org.cloudbus.cloudsim.examples.topologyrl;

public class Session {
    public final int userId;
    public final int nodeId;
    public final TierSpec tier;
    public final int startSlot;
    public final int endSlot;
    public final double computeDelayMs;
    public final double txDelayMs;

    public Session(int userId, int nodeId, TierSpec tier, int startSlot, int endSlot, double computeDelayMs, double txDelayMs) {
        this.userId = userId;
        this.nodeId = nodeId;
        this.tier = tier;
        this.startSlot = startSlot;
        this.endSlot = endSlot;
        this.computeDelayMs = computeDelayMs;
        this.txDelayMs = txDelayMs;
    }

    public double e2eDelayMs() {
        return computeDelayMs + txDelayMs;
    }
}
