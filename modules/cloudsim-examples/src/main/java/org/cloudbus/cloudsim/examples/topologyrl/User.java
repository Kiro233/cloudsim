package org.cloudbus.cloudsim.examples.topologyrl;

import java.util.List;

public class User {
    public final int userId;
    public final TierSpec tier;
    public final List<Integer> candidateNodes;
    public Integer busyUntilSlot = null;

    public User(int userId, TierSpec tier, List<Integer> candidateNodes) {
        this.userId = userId;
        this.tier = tier;
        this.candidateNodes = candidateNodes;
    }

    public boolean isIdle() {
        return busyUntilSlot == null;
    }
}
