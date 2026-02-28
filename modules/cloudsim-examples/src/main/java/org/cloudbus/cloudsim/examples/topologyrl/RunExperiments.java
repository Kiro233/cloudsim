package org.cloudbus.cloudsim.examples.topologyrl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class RunExperiments {
    public static void main(String[] args) throws IOException {
        List<Integer> lambdas = Arrays.asList(4, 8, 12, 16);
        List<Integer> seeds = Arrays.asList(0, 1, 2, 3, 4);

        SimConfig baseCfg = new SimConfig();
        List<Scheduler> schedulers = new ArrayList<>();
        schedulers.add(new BestFitScheduler());
        schedulers.add(new LocalRegressionHeuristicScheduler(baseCfg.nodeCostPerMinute(), baseCfg.localRegression));
        schedulers.add(new RandomFeasibleScheduler());
        schedulers.add(new PythonPolicyScheduler(
                baseCfg.pythonPolicyUrl,
                baseCfg.decisionTimeoutMs,
                new BestFitScheduler()
        ));

        List<String> rows = new ArrayList<>();
        rows.add("lambda_per_hour,seed,scheduler,total_arrivals,accepted,rejected,accept_rate,sla_rate,avg_delay_ms,p95_delay_ms,avg_control_delay_ms,p95_control_delay_ms,total_cost_cny,cost_per_accepted_cny,fallback_timeout,fallback_service_unavailable,fallback_illegal_action,policy_reject");

        for (int lambda : lambdas) {
            System.out.println("\n=== Lambda = " + lambda + "/hour ===");
            for (Scheduler scheduler : schedulers) {
                List<Double> acceptRates = new ArrayList<>();
                List<Double> slaRates = new ArrayList<>();
                List<Double> costPerAccepted = new ArrayList<>();
                List<Double> p95List = new ArrayList<>();
                List<Double> avgControlDelayList = new ArrayList<>();
                List<Double> p95ControlDelayList = new ArrayList<>();

                int totalFallbackTimeout = 0;
                int totalFallbackServiceUnavailable = 0;
                int totalFallbackIllegalAction = 0;
                int totalPolicyReject = 0;

                for (int seed : seeds) {
                    SimConfig cfg = new SimConfig();
                    cfg.lambdaPerHour = lambda;

                    if (scheduler instanceof PythonPolicyScheduler) {
                        ((PythonPolicyScheduler) scheduler).resetStats();
                    }

                    TopologySimulation sim = new TopologySimulation(cfg, seed);
                    SimResult r = sim.run(scheduler);

                    double acceptRate = r.totalArrivals > 0 ? ((double) r.accepted / r.totalArrivals) : 0.0;
                    double slaRate = r.accepted > 0 ? ((double) (r.accepted - r.slaViolations) / r.accepted) : 0.0;
                    double cps = r.accepted > 0 ? (r.totalCostCny / r.accepted) : 0.0;

                    int fallbackTimeout = 0;
                    int fallbackServiceUnavailable = 0;
                    int fallbackIllegalAction = 0;
                    int policyReject = 0;

                    if (scheduler instanceof PythonPolicyScheduler) {
                        PythonPolicyScheduler py = (PythonPolicyScheduler) scheduler;
                        fallbackTimeout = py.getFallbackTimeoutCount();
                        fallbackServiceUnavailable = py.getFallbackServiceUnavailableCount();
                        fallbackIllegalAction = py.getFallbackIllegalActionCount();
                        policyReject = py.getPolicyRejectCount();

                        totalFallbackTimeout += fallbackTimeout;
                        totalFallbackServiceUnavailable += fallbackServiceUnavailable;
                        totalFallbackIllegalAction += fallbackIllegalAction;
                        totalPolicyReject += policyReject;
                    }

                    acceptRates.add(acceptRate);
                    slaRates.add(slaRate);
                    costPerAccepted.add(cps);
                    p95List.add(r.p95E2eDelayMs());
                    avgControlDelayList.add(r.avgControlDelayMs());
                    p95ControlDelayList.add(r.p95ControlDelayMs());

                    rows.add(String.format(
                            "%d,%d,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d",
                            lambda,
                            seed,
                            scheduler.name(),
                            r.totalArrivals,
                            r.accepted,
                            r.rejected,
                            acceptRate,
                            slaRate,
                            r.avgE2eDelayMs(),
                            r.p95E2eDelayMs(),
                            r.avgControlDelayMs(),
                            r.p95ControlDelayMs(),
                            r.totalCostCny,
                            cps,
                            fallbackTimeout,
                            fallbackServiceUnavailable,
                            fallbackIllegalAction,
                            policyReject
                    ));
                }

                System.out.println(String.format(
                        "[%s] accept=%.3f±%.3f, sla=%.3f±%.3f, cost/accepted=%.4f±%.4f CNY, p95=%.3f±%.3f ms, ctrl_avg=%.3f±%.3f ms, ctrl_p95=%.3f±%.3f ms, fb_timeout=%d, fb_down=%d, fb_illegal=%d, policy_reject=%d",
                        scheduler.name(),
                        mean(acceptRates), std(acceptRates),
                        mean(slaRates), std(slaRates),
                        mean(costPerAccepted), std(costPerAccepted),
                        mean(p95List), std(p95List),
                        mean(avgControlDelayList), std(avgControlDelayList),
                        mean(p95ControlDelayList), std(p95ControlDelayList),
                        totalFallbackTimeout,
                        totalFallbackServiceUnavailable,
                        totalFallbackIllegalAction,
                        totalPolicyReject
                ));
            }
        }

        Path out = Paths.get("experiments", "topology_sim", "results_java.csv");
        Files.createDirectories(out.getParent());
        Files.write(out, rows, StandardCharsets.UTF_8);
        System.out.println("\nWrote results to " + out.toAbsolutePath());
    }

    private static double mean(List<Double> values) {
        if (values.isEmpty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.size();
    }

    private static double std(List<Double> values) {
        if (values.size() <= 1) {
            return 0.0;
        }
        double m = mean(values);
        double s = 0.0;
        for (double v : values) {
            double d = v - m;
            s += d * d;
        }
        return Math.sqrt(s / values.size());
    }
}
