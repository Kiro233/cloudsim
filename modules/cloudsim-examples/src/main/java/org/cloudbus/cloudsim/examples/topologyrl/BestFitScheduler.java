package org.cloudbus.cloudsim.examples.topologyrl;

import java.util.List;

public class BestFitScheduler implements Scheduler {
    @Override
    public String name() {
        return "best_fit";
    }

    @Override
    public SchedulingDecision selectNode(User user, List<NodeState> nodes) {
        Integer bestNodeId = null;
        double bestRemaining = Double.POSITIVE_INFINITY;

        for (int nodeId : user.candidateNodes) {
            NodeState node = nodes.get(nodeId);
            if (!node.canAccept(user.tier.name)) {
                continue;
            }

            double step = "L".equals(user.tier.name) ? 1.0 / 3.0 : 1.0 / 2.0;
            double afterRemaining = node.remainingCapacity() - step;
            if (afterRemaining < bestRemaining) {
                bestRemaining = afterRemaining;
                bestNodeId = nodeId;
            }
        }

        return bestNodeId == null ? SchedulingDecision.reject() : SchedulingDecision.accept(bestNodeId);
    }
}
