package org.cloudbus.cloudsim.examples;

/*
 * 标题:        CloudSim 工具包
 * 描述:        用于云计算建模与仿真的 CloudSim（云仿真）工具包
 *               （Clouds，即云环境）
 * 许可证:      GPL - 详见 http://www.gnu.org/copyleft/gpl.html
 *
 * 版权 (c) 2009, The University of Melbourne, Australia
 */

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;

/**
 * 一个简单示例，演示如何创建仅包含一个主机的数据中心，
 * 并在其上运行一个 Cloudlet（任务）。
 */
public class CloudSimExample1 {
	public static DatacenterBroker broker;

	/** Cloudlet（任务）列表。 */
	private static List<Cloudlet> cloudletList;
	/** 虚拟机列表。 */
	private static List<Vm> vmlist;

	/**
	 * main() 方法：运行本示例。
	 *
	 * @param args 命令行参数
	 */
	public static void main(String[] args) {
		Log.println("Starting CloudSimExample1...");

		try {
			// 第一步：初始化 CloudSim 包。在创建任何实体之前必须先调用。
			int num_user = 1; // 云用户数量
			Calendar calendar = Calendar.getInstance(); // 日历对象，其字段已初始化为当前日期和时间。
 			boolean trace_flag = false; // 是否追踪事件

			/* 注释开始 - Dinesh Bhagwat 
			 * 初始化 CloudSim 库。
			 * init() 调用 initCommonVariable()，而 initCommonVariable() 又会调用 initialize()
			 * （这三个方法都定义在 CloudSim.java 中）。
			 * initialize() 会创建两个集合：
			 *   - 一个 SimEntity 对象的 ArrayList（名为 entities，表示仿真实体）；
			 *   - 一个 LinkedHashMap（名为 entitiesByName，表示同一批仿真实体的映射），
			 *     其中每个 SimEntity 的名称作为键。
			 * initialize() 会创建两个队列：
			 *   - 一个 SimEvent 的队列（future，未来事件队列）；
			 *   - 另一个 SimEvent 的队列（deferred，延迟事件队列）。
			 * initialize() 会创建一个以整数为键的 Predicate（谓词）HashMap，
			 *   这些谓词用于从延迟队列中选择特定事件。
			 * initialize() 会把仿真时钟设为 0，并把运行标志 running（布尔值）置为 false。
			 * 一旦 initialize() 返回（此时我们位于 initCommonVariable() 方法中），
			 *   就会创建一个 CloudSimShutDown 实例（继承自 SimEntity），
			 *   其参数为：numuser 为 1，名称为 CloudSimShutDown，id 为 -1，状态为 RUNNABLE。
			 * 然后该新实体会被加入仿真环境。
			 * 在被加入仿真环境时，它的 id 会从原来的 -1 变为 0。
			 * 两个集合 entities 和 entitiesByName 会用该 SimEntity 进行更新。
			 * 此时 shutdownId（默认值为 -1）变为 0。
			 * 当 initCommonVariable() 返回后（此时我们位于 init() 方法中），
			 *   会创建一个 CloudInformationService 实例（同样继承自 SimEntity），
			 *   其名称为 CloudInformationService，id 为 -1，状态为 RUNNABLE。
			 * 然后该新实体也会被加入仿真环境。
			 * 在被加入仿真环境时，该 SimEntity 的 id 会从原来的 -1 变为 1（下一个可用 id）。
			 * 两个集合 entities 和 entitiesByName 再次被该 SimEntity 更新。
			 * 此时 cisId（默认值为 -1）变为 1。
			 * 注释结束 - Dinesh Bhagwat 
			 */
			CloudSim.init(num_user, calendar, trace_flag);

			// 第二步：创建数据中心（Datacenter）。
			// 在 CloudSim 中，数据中心是资源提供者。
			// 至少需要创建一个数据中心才能运行 CloudSim 仿真。
			Datacenter datacenter0 = createDatacenter("Datacenter_0");

			// 第三步：创建 Broker（代理）。
			broker = new DatacenterBroker("Broker");
			int brokerId = broker.getId();

			// 第四步：创建一个虚拟机（VM）。
			vmlist = new ArrayList<>();

			// 虚拟机参数描述
			int vmid = 0;
			int mips = 1000;
			long size = 10000; // 镜像大小（MB）
			int ram = 512; // 虚拟机内存（MB）
			long bw = 1000;
			int pesNumber = 1; // CPU 数量（处理单元个数）
			String vmm = "Xen"; // 虚拟机监控器（VMM）名称

			// 创建虚拟机
			Vm vm = new Vm(vmid, brokerId, mips, pesNumber, ram, bw, size, vmm, new CloudletSchedulerTimeShared());

			// 将该虚拟机加入虚拟机列表
			vmlist.add(vm);

			// 向 Broker 提交虚拟机列表
			broker.submitGuestList(vmlist);

			// 第五步：创建一个 Cloudlet（任务）。
			cloudletList = new ArrayList<>();

			// Cloudlet 参数属性
			int id = 0;
			long length = 400000;
			long fileSize = 300;
			long outputSize = 300;
			UtilizationModel utilizationModel = new UtilizationModelFull();

			Cloudlet cloudlet = new Cloudlet(id, length, pesNumber, fileSize,
                                        outputSize, utilizationModel, utilizationModel, 
                                        utilizationModel);
			cloudlet.setUserId(brokerId);
			cloudlet.setGuestId(vmid);

			// 将该 Cloudlet 加入列表
			cloudletList.add(cloudlet);

			// 向 Broker 提交 Cloudlet 列表
			broker.submitCloudletList(cloudletList);

			// 第六步：启动仿真
			CloudSim.startSimulation();

			CloudSim.stopSimulation();

			// 最后一步：仿真结束后打印结果
			List<Cloudlet> newList = broker.getCloudletReceivedList();
			printCloudletList(newList);

			Log.println("CloudSimExample1 finished!");
		} catch (Exception e) {
			e.printStackTrace();
			Log.println("Unwanted errors happen");
		}
	}

	/**
	 * 创建数据中心。
	 *
	 * @param name 数据中心名称
	 *
	 * @return 创建好的数据中心对象
	 */
	private static Datacenter createDatacenter(String name) {

		// 创建数据中心（PowerDatacenter）需要以下步骤：
		// 1. 创建一个列表，用于存放主机（Host）
		List<Host> hostList = new ArrayList<>();

		// 2. 一台机器（Host）包含一个或多个处理单元（PE，即 CPU/Core）。
		//    在本示例中，仅包含一个核心。
		List<Pe> peList = new ArrayList<>();

		int mips = 1000;

		// 3. 创建处理单元（PE），并加入到列表中。
		peList.add(new Pe(new PeProvisionerSimple(mips))); // 需要存储 PE 的 id 和 MIPS 额定值

		// 4. 使用 id 和 PE 列表创建 Host，并把它加入到 Host 列表中。
		int ram = 2048; // 主机内存（MB）
		long storage = 1000000; // 主机存储容量
		int bw = 10000;

		hostList.add(
			new Host(
				new RamProvisionerSimple(ram),
				new BwProvisionerSimple(bw),
				storage,
				peList,
				new VmSchedulerTimeShared(peList)
			)
		); // 这就是我们的一台主机（机器）

		// 5. 创建 DatacenterCharacteristics 对象，用于存储数据中心的属性：
		//    架构、操作系统、主机列表、分配策略（时间共享或空间共享）、时区以及价格（G$/Pe 时间单位）。
		String arch = "x86"; // 系统架构
		String os = "Linux"; // 操作系统
		String vmm = "Xen";
		double time_zone = 10.0; // 此资源所在的时区
		double cost = 3.0; // 使用该资源处理能力的成本
		double costPerMem = 0.05; // 使用该资源内存的成本
		double costPerStorage = 0.001; // 使用该资源存储的成本
		double costPerBw = 0.0; // 使用该资源带宽的成本
		LinkedList<Storage> storageList = new LinkedList<>(); // 当前不添加 SAN 设备

		DatacenterCharacteristics characteristics = new DatacenterCharacteristics(
				arch, os, vmm, hostList, time_zone, cost, costPerMem,
				costPerStorage, costPerBw);

		// 6. 最后，创建一个 PowerDatacenter 对象。
		Datacenter datacenter = null;
		try {
			datacenter = new Datacenter(name, characteristics, new VmAllocationPolicySimple(hostList), storageList, 0);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return datacenter;
	}

	/**
	 * 打印 Cloudlet（任务）执行结果。
	 *
	 * @param list Cloudlet 列表
	 */
	private static void printCloudletList(List<Cloudlet> list) {
		int size = list.size();
		Cloudlet cloudlet;

		String indent = "    ";
		Log.println();
		Log.println("========== OUTPUT ==========");
		Log.println("Cloudlet ID" + indent + "STATUS" + indent
				+ "Data center ID" + indent + "VM ID" + indent + "Time" + indent
				+ "Start Time" + indent + "Finish Time");

		DecimalFormat dft = new DecimalFormat("###.##");
		for (Cloudlet value : list) {
			cloudlet = value;
			Log.print(indent + cloudlet.getCloudletId() + indent + indent);

			if (cloudlet.getStatus() == Cloudlet.CloudletStatus.SUCCESS) {
				Log.print("SUCCESS");

				Log.println(indent + indent + cloudlet.getResourceId()
						+ indent + indent + indent + cloudlet.getGuestId()
						+ indent + indent
						+ dft.format(cloudlet.getActualCPUTime()) + indent
						+ indent + dft.format(cloudlet.getExecStartTime())
						+ indent + indent
						+ dft.format(cloudlet.getExecFinishTime()));
			}
		}
	}
}