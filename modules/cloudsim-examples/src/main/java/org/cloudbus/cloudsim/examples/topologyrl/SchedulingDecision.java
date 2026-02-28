package org.cloudbus.cloudsim.examples.topologyrl;

public class SchedulingDecision {
    public final boolean accepted;
    public final Integer nodeId;

    public SchedulingDecision(boolean accepted, Integer nodeId) {
        this.accepted = accepted;
        this.nodeId = nodeId;
    }

    public static SchedulingDecision reject() {
        return new SchedulingDecision(false, null);
    }

    public static SchedulingDecision accept(int nodeId) {
        return new SchedulingDecision(true, nodeId);
    }
}
