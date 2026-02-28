package org.cloudbus.cloudsim.examples.topologyrl;

import java.util.HashSet;
import java.util.Set;

public class NodeState {
    public final int nodeId;
    public int nL = 0;
    public int nH = 0;
    public final Set<Integer> activeUserIds = new HashSet<>();

    public NodeState(int nodeId) {
        this.nodeId = nodeId;
    }

    public boolean canAccept(String tierName) {
        int nextL = nL + ("L".equals(tierName) ? 1 : 0);
        int nextH = nH + ("H".equals(tierName) ? 1 : 0);
        return (nextL / 3.0 + nextH / 2.0) <= 1.0 + 1e-9;
    }

    public void addUser(int userId, String tierName) {
        if ("L".equals(tierName)) {
            nL += 1;
        } else {
            nH += 1;
        }
        activeUserIds.add(userId);
    }

    public void removeUser(int userId, String tierName) {
        if ("L".equals(tierName)) {
            nL -= 1;
        } else {
            nH -= 1;
        }
        activeUserIds.remove(userId);
    }

    public double remainingCapacity() {
        return 1.0 - (nL / 3.0 + nH / 2.0);
    }

    public boolean isEnabled() {
        return !activeUserIds.isEmpty();
    }
}
