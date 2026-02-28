package org.cloudbus.cloudsim.examples.topologyrl;

import java.util.List;

public interface Scheduler {
    String name();

    SchedulingDecision selectNode(User user, List<NodeState> nodes);
}
