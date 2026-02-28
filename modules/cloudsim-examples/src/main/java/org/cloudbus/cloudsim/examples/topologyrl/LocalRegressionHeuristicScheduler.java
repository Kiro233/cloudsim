package org.cloudbus.cloudsim.examples.topologyrl;

import java.util.List;

public class LocalRegressionHeuristicScheduler implements Scheduler {
    private final double nodeCostCnyPerMinute;
    private final SimConfig.LocalRegressionParams p;

    public LocalRegressionHeuristicScheduler(double nodeCostCnyPerMinute, SimConfig.LocalRegressionParams params) {
        this.nodeCostCnyPerMinute = nodeCostCnyPerMinute;
        this.p = params;
    }

    @Override
    public String name() {
        return "local_regression_heuristic";
    }

    @Override
    public SchedulingDecision selectNode(User user, List<NodeState> nodes) {
        Integer bestNodeId = null;
        double bestScore = Double.POSITIVE_INFINITY;
        int tierIndicator = "L".equals(user.tier.name) ? 0 : 1;

        for (int nodeId : user.candidateNodes) {
            NodeState node = nodes.get(nodeId);
            if (!node.canAccept(user.tier.name)) {
                continue;
            }

            double rho = node.nL / 3.0 + node.nH / 2.0;
            double q = node.activeUserIds.size();
            double dHat = p.a0 + p.a1 * rho + p.a2 * q + p.a3 * tierIndicator + p.a4 * user.tier.txDelayMs;

            double deltaCost = node.isEnabled() ? 0.0 : nodeCostCnyPerMinute;
            double frag = fragAfterPlacement(node, user.tier.name);

            double score = p.alpha * dHat + p.beta * deltaCost + p.gamma * frag;
            if (score < bestScore) {
                bestScore = score;
                bestNodeId = nodeId;
            }
        }

        return bestNodeId == null ? SchedulingDecision.reject() : SchedulingDecision.accept(bestNodeId);
    }

    private double fragAfterPlacement(NodeState node, String tierName) {
        int nextL = node.nL + ("L".equals(tierName) ? 1 : 0);
        int nextH = node.nH + ("H".equals(tierName) ? 1 : 0);
        return Math.abs(nextL / 3.0 - nextH / 2.0);
    }
}
