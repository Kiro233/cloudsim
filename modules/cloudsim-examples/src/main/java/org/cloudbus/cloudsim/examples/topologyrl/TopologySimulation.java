package org.cloudbus.cloudsim.examples.topologyrl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class TopologySimulation {
    private final SimConfig cfg;
    private final Random rng;

    private final List<User> users;
    private final List<NodeState> nodes;
    private final Map<Integer, Session> userToSession = new HashMap<>();

    public TopologySimulation(SimConfig cfg, long seed) {
        this.cfg = cfg;
        this.rng = new Random(seed);
        this.nodes = buildNodes(cfg.numNodes);
        this.users = buildUsers();
    }

    private List<NodeState> buildNodes(int numNodes) {
        List<NodeState> out = new ArrayList<>();
        for (int i = 0; i < numNodes; i++) {
            out.add(new NodeState(i));
        }
        return out;
    }

    private List<User> buildUsers() {
        List<User> out = new ArrayList<>();
        List<Integer> allNodeIds = new ArrayList<>();
        for (int i = 0; i < cfg.numNodes; i++) {
            allNodeIds.add(i);
        }

        for (int uid = 0; uid < cfg.numPotentialUsers; uid++) {
            TierSpec tier = rng.nextDouble() < cfg.tierLRatio ? TierSpec.L : TierSpec.H;
            List<Integer> candidates = sampleWithoutReplacement(allNodeIds, cfg.avgCandidateNodes);
            out.add(new User(uid, tier, candidates));
        }
        return out;
    }

    public SimResult run(Scheduler scheduler) {
        SimResult result = new SimResult(scheduler.name(), cfg.lambdaPerHour);

        for (int slot = 0; slot < cfg.totalSlots(); slot++) {
            endFinishedSessions(slot);

            int enabledNodes = 0;
            for (NodeState node : nodes) {
                if (node.isEnabled()) {
                    enabledNodes += 1;
                }
            }
            result.totalCostCny += enabledNodes * cfg.nodeCostPerMinute() * cfg.slotMinutes;

            int arrivals = poissonArrivalsPerSlot();
            List<User> idleUsers = getIdleUsers();
            int servedArrivals = Math.min(arrivals, idleUsers.size());

            for (int k = 0; k < servedArrivals; k++) {
                result.totalArrivals += 1;
                User user = drawAndRemoveRandom(idleUsers);

                long t0 = System.nanoTime();
                SchedulingDecision decision = scheduler.selectNode(user, nodes);
                long t1 = System.nanoTime();
                result.addControlDelayMs((t1 - t0) / 1_000_000.0);
                if (!decision.accepted || decision.nodeId == null) {
                    result.rejected += 1;
                    continue;
                }

                NodeState node = nodes.get(decision.nodeId);
                if (!node.canAccept(user.tier.name)) {
                    result.rejected += 1;
                    continue;
                }

                double computeDelayMs = computeDelayMs(node, user.tier);
                double e2eDelay = computeDelayMs + user.tier.txDelayMs;

                int sessionSlots = sampleSessionSlots();
                Session session = new Session(
                        user.userId,
                        node.nodeId,
                        user.tier,
                        slot,
                        slot + sessionSlots,
                        computeDelayMs,
                        user.tier.txDelayMs
                );

                userToSession.put(user.userId, session);
                user.busyUntilSlot = session.endSlot;
                node.addUser(user.userId, user.tier.name);

                result.accepted += 1;
                result.addDelay(e2eDelay);
                if (e2eDelay > cfg.slaMs) {
                    result.slaViolations += 1;
                }
            }
        }

        return result;
    }

    private void endFinishedSessions(int currentSlot) {
        List<Integer> endingUserIds = new ArrayList<>();
        for (Map.Entry<Integer, Session> entry : userToSession.entrySet()) {
            if (entry.getValue().endSlot <= currentSlot) {
                endingUserIds.add(entry.getKey());
            }
        }

        for (Integer uid : endingUserIds) {
            Session s = userToSession.remove(uid);
            nodes.get(s.nodeId).removeUser(uid, s.tier.name);
            users.get(uid).busyUntilSlot = null;
        }
    }

    private double computeDelayMs(NodeState node, TierSpec tier) {
        double wNode = node.nL * TierSpec.L.computeTflopsS + node.nH * TierSpec.H.computeTflopsS + tier.computeTflopsS;
        return (wNode / cfg.gpuTflops) * 1000.0;
    }

    private int poissonArrivalsPerSlot() {
        double ratePerSlot = cfg.lambdaPerHour * (cfg.slotMinutes / 60.0);
        double l = Math.exp(-ratePerSlot);
        int k = 0;
        double p = 1.0;
        do {
            k++;
            p *= rng.nextDouble();
        } while (p > l);
        return k - 1;
    }

    private int sampleSessionSlots() {
        double hours = cfg.sessionHoursMin + rng.nextDouble() * (cfg.sessionHoursMax - cfg.sessionHoursMin);
        return Math.max(1, (int) (hours * 60 / cfg.slotMinutes));
    }

    private List<User> getIdleUsers() {
        List<User> idle = new ArrayList<>();
        for (User u : users) {
            if (u.isIdle()) {
                idle.add(u);
            }
        }
        return idle;
    }

    private User drawAndRemoveRandom(List<User> usersList) {
        int idx = rng.nextInt(usersList.size());
        return usersList.remove(idx);
    }

    private List<Integer> sampleWithoutReplacement(List<Integer> src, int k) {
        List<Integer> copied = new ArrayList<>(src);
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < k && !copied.isEmpty(); i++) {
            int idx = rng.nextInt(copied.size());
            out.add(copied.remove(idx));
        }
        return out;
    }
}
