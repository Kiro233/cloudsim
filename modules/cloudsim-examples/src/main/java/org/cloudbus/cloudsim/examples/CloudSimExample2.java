/*
 * 标题:        CloudSim 工具包
 * 描述:        CloudSim（云仿真）工具包，用于云的建模和仿真
 * 许可证:      GPL - http://www.gnu.org/copyleft/gpl.html
 *
 * 版权所有 (c) 2009, 墨尔本大学, 澳大利亚
 */

package org.cloudbus.cloudsim.examples;


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
 * 一个简单的示例，展示如何创建
 * 一个包含一个主机的数据中心，并在其上运行两个
 * 云任务。云任务在具有相同MIPS要求的
 * 虚拟机中运行。这些云任务将花费相同的时间
 * 来完成执行。
 */
public class CloudSimExample2 {
	public static DatacenterBroker broker;

	/** 云任务列表。 */
	private static List<Cloudlet> cloudletList;

	/** 虚拟机列表。 */
	private static List<Vm> vmlist;

	/**
	 * 创建 main() 方法来运行此示例
	 */
	public static void main(String[] args) {

		Log.println("Starting CloudSimExample2...");

	        try {
	        	// 第一步：初始化 CloudSim 包。在创建任何实体之前应该调用它。
	            	int num_user = 1;   // 云用户数量
	            	Calendar calendar = Calendar.getInstance();
	            	boolean trace_flag = false;  // 表示跟踪事件

	            	// 初始化 CloudSim 库
	            	CloudSim.init(num_user, calendar, trace_flag);

	            	// 第二步：创建数据中心
	            	// 数据中心是 CloudSim 中的资源提供者。我们需要至少一个数据中心来运行 CloudSim 仿真
	            	Datacenter datacenter0 = createDatacenter("Datacenter_0");

	            	// 第三步：创建代理
			broker = new DatacenterBroker("Broker");
			int brokerId = broker.getId();

	            	// 第四步：创建虚拟机
	            	vmlist = new ArrayList<>();

	            	// 虚拟机描述
	            	int mips = 250;
	            	long size = 10000; // 镜像大小 (MB)
	            	int ram = 512; // 虚拟机内存 (MB)
	            	long bw = 1000;
	            	int pesNumber = 1; // CPU 数量
	            	String vmm = "Xen"; // 虚拟机监视器名称

	            	// 创建两个虚拟机
			Vm vm1 = new Vm(brokerId, mips, pesNumber, ram, bw, size, vmm, new CloudletSchedulerTimeShared());
			Vm vm2 = new Vm(brokerId, mips, pesNumber, ram, bw, size, vmm, new CloudletSchedulerTimeShared());

	            	// 将虚拟机添加到虚拟机列表
	            	vmlist.add(vm1);
	            	vmlist.add(vm2);

	            	// 向代理提交虚拟机列表
	            	broker.submitGuestList(vmlist);

	            	// 第五步：创建两个云任务
	            	cloudletList = new ArrayList<>();

	            	// 云任务属性
	            	pesNumber=1;
	            	long length = 250000;
	            	long fileSize = 300;
	            	long outputSize = 300;
	            	UtilizationModel utilizationModel = new UtilizationModelFull();

			Cloudlet cloudlet1 = new Cloudlet(length, pesNumber, fileSize, outputSize, utilizationModel, utilizationModel, utilizationModel);
	            	cloudlet1.setUserId(brokerId);

			Cloudlet cloudlet2 = new Cloudlet(length, pesNumber, fileSize, outputSize, utilizationModel, utilizationModel, utilizationModel);
	            	cloudlet2.setUserId(brokerId);

	            	// 将云任务添加到列表
	            	cloudletList.add(cloudlet1);
	            	cloudletList.add(cloudlet2);

	            	// 向代理提交云任务列表
	            	broker.submitCloudletList(cloudletList);


	            	// 将云任务绑定到虚拟机。这样，代理
	            	// 将仅向特定虚拟机提交绑定的云任务
	            	broker.bindCloudletToVm(cloudlet1.getCloudletId(),vm1.getId());
	            	broker.bindCloudletToVm(cloudlet2.getCloudletId(),vm2.getId());

	            	// 第六步：开始仿真
	            	CloudSim.startSimulation();


	            	// 最后一步：仿真结束时打印结果
	            	List<Cloudlet> newList = broker.getCloudletReceivedList();

	            	CloudSim.stopSimulation();

	            	printCloudletList(newList);

	            	Log.println("CloudSimExample2 完成！");
	        }
	        catch (Exception e) {
	            e.printStackTrace();
	            Log.println("仿真因意外错误而终止");
	        }
	    }

		private static Datacenter createDatacenter(String name){

        // 创建 PowerDatacenter 需要以下步骤：
        // 1. 我们需要创建一个列表来存储
        //    我们的机器
        List<Host> hostList = new ArrayList<>();

        // 2. 一台机器包含一个或多个 PE（处理元素）或 CPU/核心。
        // 在这个例子中，它只有一个核心。
        List<Pe> peList = new ArrayList<>();

        int mips = 1000;

        // 3. 创建 PE 并将它们添加到列表中。
		peList.add(new Pe(new PeProvisionerSimple(mips))); // 需要存储 Pe ID 和 MIPS 评级

        // 4. 创建主机，指定其 ID 和 PE 列表，并将它们添加到机器列表中
        int ram = 2048; // 主机内存 (MB)
        long storage = 1000000; // 主机存储
        int bw = 10000;

        hostList.add(
            		new Host(
            			new RamProvisionerSimple(ram),
            			new BwProvisionerSimple(bw),
            			storage,
            			peList,
            			new VmSchedulerTimeShared(peList)
            		)
            	); // 这是我们的机器


        // 5. 创建一个 DatacenterCharacteristics 对象，存储数据中心的属性：
        //    架构、操作系统、机器列表、分配策略（时间共享或空间共享）、
        //    时区及其价格（G$/Pe 时间单位）。
        String arch = "x86";      // 系统架构
        String os = "Linux";          // 操作系统
        String vmm = "Xen";
        double time_zone = 10.0;         // 此资源所在的时区
        double cost = 3.0;              // 使用此资源处理的成本
        double costPerMem = 0.05;      // 使用此资源内存的成本
        double costPerStorage = 0.001;  // 使用此资源存储的成本
        double costPerBw = 0.0;         // 使用此资源带宽的成本
        LinkedList<Storage> storageList = new LinkedList<>();  // 我们现在不添加 SAN 设备

        DatacenterCharacteristics characteristics = new DatacenterCharacteristics(
                arch, os, vmm, hostList, time_zone, cost, costPerMem, costPerStorage, costPerBw);


        // 6. 最后，我们需要创建一个 PowerDatacenter 对象。
        Datacenter datacenter = null;
        try {
            datacenter = new Datacenter(name, characteristics, new VmAllocationPolicySimple(hostList), storageList, 0);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return datacenter;
    }

	    /**
     * 打印云任务对象
     * @param list 云任务列表
     */
    private static void printCloudletList(List<Cloudlet> list) {
        int size = list.size();
        Cloudlet cloudlet;

        String indent = "    ";
        Log.println();
        Log.println("========== 输出 ==========");
        Log.println("云任务 ID" + indent + "状态" + indent +
                "数据中心 ID" + indent + "虚拟机 ID" + indent + "时间" + indent + "开始时间" + indent + "完成时间");

        DecimalFormat dft = new DecimalFormat("###.##");
		for (Cloudlet value : list) {
			cloudlet = value;
			Log.print(indent + cloudlet.getCloudletId() + indent + indent);

			if (cloudlet.getStatus() == Cloudlet.CloudletStatus.SUCCESS) {
				Log.print("成功");

				Log.println(indent + indent + cloudlet.getResourceId() + indent + indent + indent + cloudlet.getGuestId() +
						indent + indent + dft.format(cloudlet.getActualCPUTime()) + indent + indent + dft.format(cloudlet.getExecStartTime()) +
						indent + indent + dft.format(cloudlet.getExecFinishTime()));
			}
		}

    }
}
