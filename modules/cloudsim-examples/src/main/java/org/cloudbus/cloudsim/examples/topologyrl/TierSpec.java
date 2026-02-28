package org.cloudbus.cloudsim.examples.topologyrl;

public class TierSpec {
    public final String name;
    public final double computeTflopsS;
    public final long lengthMi;
    public final double downlinkMb;
    public final double txDelayMs;

    public TierSpec(String name, double computeTflopsS, long lengthMi, double downlinkMb, double txDelayMs) {
        this.name = name;
        this.computeTflopsS = computeTflopsS;
        this.lengthMi = lengthMi;
        this.downlinkMb = downlinkMb;
        this.txDelayMs = txDelayMs;
    }

    public static final TierSpec L = new TierSpec("L", 0.31758, 317_580, 0.0958, 0.958);
    public static final TierSpec H = new TierSpec("H", 0.59362, 593_620, 0.1916, 1.916);
}
