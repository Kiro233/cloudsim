package org.cloudbus.cloudsim.examples.topologyrl;

import java.util.List;

public class RandomFeasibleScheduler implements Scheduler {
    @Override
    public String name() {
        return "random_feasible";
    }

    @Override
    public SchedulingDecision selectNode(User user, List<NodeState> nodes) {
        for (int nodeId : user.candidateNodes) {
            if (nodes.get(nodeId).canAccept(user.tier.name)) {
                return SchedulingDecision.accept(nodeId);
            }
        }
        return SchedulingDecision.reject();
    }
}
