# TB6-R5-RevA1 机型包 · 目录结构说明

本文件逐项说明 `RevA1-sim/TB6-R5-RevA1/` 目录下**每个文件夹与每个文件**的名称、作用和数据含义。

- 这是什么：**SolidWorks 插件 [sw_urdf_exporter](http://wiki.ros.org/sw_urdf_exporter) 自动导出的 ROS URDF 机型包**。
  源装配体是 `7260501-000000-001 TB6-R5-RevA1.SLDASM`，导出时间见 `export.log`（最后一次成功导出：2026-06-02 17:09）。
- 在本仓库里的地位：`RevA1-sim` 的**仿真唯一数据源**（真值来源）。
  `revA1_spec.py` 的 `PACKAGE_ROOT` 默认指向本目录，`convert_urdf_to_mjcf.py` / `model_import.py`
  读取这里的 URDF 与 `meshes/*.STL` 生成 MuJoCo 模型；**改了这里的 URDF 就要重跑一次转换脚本**。
- 命名来源：`7260501-000000-001` 是物料/项目号，`TB6-R5` 是机型，`RevA1` 是版本号，`URDF` 是导出格式。

---

## 1. 目录树

```text
TB6-R5-RevA1/
├── CMakeLists.txt                              # catkin 构建脚本：安装 config/launch/meshes/urdf
├── package.xml                                 # ROS 包清单（包名 = 7260501-000000-001 TB6-R5-RevA1-URDF）
├── RDM.xml                                     # EtherCAT 从站描述文件（Techservo RDM 驱动器，非 ROS 文件）
├── export.log                                  # sw_urdf_exporter 的导出日志（记录了每个 STL/URDF 的生成过程）
├── README.md                                   # 本文件
├── config/
│   └── joint_names_7260501-000000-001 TB6-R5-RevA1-URDF.yaml   # 控制器关节名列表
├── launch/
│   ├── display.launch                          # RViz + joint_state_publisher_gui 显示模型
│   └── gazebo.launch                           # 在 Gazebo 空世界里生成该模型
├── meshes/                                     # 7 个连杆的 3D 网格（二进制 STL）
│   ├── base_link.STL                           # 基座
│   ├── Link1.STL ~ Link5.STL                   # 第 1~5 轴连杆
│   └── ee_Link.STL                             # 末端法兰（工具接口）
└── urdf/
    ├── 7260501-000000-001 TB6-R5-RevA1-URDF.urdf   # 机器人描述文件（links/joints/惯量/可视化/碰撞）
    └── 7260501-000000-001 TB6-R5-RevA1-URDF.csv    # 同一份数据的电子表格版（SolidWorks 侧可回填/复用）
```

---

## 2. 根目录文件

### `package.xml` —— ROS 包清单
- `<name>`：`7260501-000000-001 TB6-R5-RevA1-URDF`（就是"包名"，URDF 里的 `package://<包名>/meshes/...` 引用它）。
- `<version>`：`1.0.0`；`<license>`：BSD（模板默认值）。
- 依赖声明：`roslaunch`、`robot_state_publisher`、`rviz`、`joint_state_publisher_gui`、`gazebo`
  —— 正好对应 `launch/` 里两个 launch 文件用到的节点。
- `<author>` / `<maintainer>` 仍是模板占位符 `TODO`，说明这是导出器生成的原始包。

### `CMakeLists.txt` —— catkin 构建脚本
- `project(7260501-000000-001 TB6-R5-RevA1-URDF)`
- 循环 `config launch meshes urdf` 四个目录，把它们安装到
  `${CATKIN_PACKAGE_SHARE_DESTINATION}`（即 `share/<包名>/`）。
- 纯数据包，没有编译目标，只做"安装/拷贝"。

### `export.log` —— 导出日志（约 1.7 MB，1397 行）
- 由 SolidWorks 插件 `sw_urdf_exporter` 写出（`Logger.cs` / `ExportHelper.cs` / `CSVImportExport.cs` 等模块打印）。
- 记录了插件版本（Commit `1.6.0-4-g7f85cfe`，Build `1.6.7995.38578`）、打开装配体、
  枚举坐标系（CoordSys）/参考轴（RefAxis）、隐藏/恢复零件、逐个导出 STL、
  最后写出 URDF 与 CSV 的完整过程。
- 日志里能看到**本次导出对应的 7 个连杆与它们所用的坐标系**：
  `base_link → frame1(Origin_global)`、`Link1 → frame2`、`Link2 → frame3`、`Link3 → frame4`、
  `Link4 → frame5`、`Link5 → frame6`、`ee_Link → frame7`（见 `urdf/*.csv` 的 `Coordinate System` 列）。
- 日志中还留有多次"试导出"的记录（输出到 `...REV`、`...RevA1_urdf`、`...RevA1-urdf`、
  `...RevA1-URDF` 等不同目录），**只有 `...RevA1-URDF` 这一份被拷进本目录**。
- 文件末尾有 7 条 `ERROR CSVImportExport.cs: 130 - The following columns were not written to the CSV:
  Link.Joint.ChildLink.link`：属于导出器的已知无害告警（"子连杆"这一列没写进 CSV），
  URDF 本身不受影响。

### `RDM.xml` —— EtherCAT 从站描述（ESI 文件，约 296 KB）
- 它**不是 ROS 文件**，而是 EtherCAT 主站（如 TwinCAT/IGH）用的**设备描述 XML**：
  根节点 `<EtherCATInfo Version="1.2">`，厂商 `<Vendor><Id>171</Id><Name>Techservo</Name>`，
  设备 `<Type ProductCode="#x1030" RevisionNo="#x00010003">RDM</Type>`，器件组 `Drive`。
- 内容：`<Profile><ProfileNo>402</ProfileNo>`（CANopen **CiA402 伺服驱动协议**）、
  PDO/同步管理器映射（`Sm`：`MBoxOut #x1000` / `MBoxIn #x1100` / `Outputs #x1200` / `Inputs #x1800`）、
  大量对象字典条目（如 `CANopen Vendor ID`、`Sync Manager 2, PDO mapping n`）、
  邮箱与 EEPROM 配置（`Eeprom ByteSize 4096`）。用于 **PDO 映射 / 对象字典 / 过程数据的标准化描述**。
- 它是 ESI 规范文件，因此内容里嵌了大量图标（`ImageData16x14` 的十六进制位图），这是文件显得很大的原因。
- 在本仓库中的作用：**驱动器参数与通信对象的参考文档**（真机联调、PDO 配置时查表用）。
  代码里目前没有任何脚本引用它；放在这里的意义是"机型包 = 机械（URDF/STL）+ 电气（驱动器 ESI）"齐备。
- 修改时间（2026-09-20）晚于其他文件，说明它是在导出之后被单独补进来的。

---

## 3. `config/` —— 配置目录

### `config/joint_names_7260501-000000-001 TB6-R5-RevA1-URDF.yaml`
```yaml
controller_joint_names: ['', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', ]
```
- **含义**：供 `joint_state_publisher`（ROS 1）读取的"控制器关节名"列表，即按顺序声明本机有
  `joint1`~`joint6` 六个可控关节。
- 第一个元素为空字符串，是导出器/工具链的占位写法；文件名中的
  `7260501-000000-001 TB6-R5-RevA1-URDF` 就是包名（部分工具会按 `<前缀>_<包名>.yaml` 约定查找）。
- 与 URDF 的一致性：URDF 里 `joint1`~`joint6` 全部是 `revolute`（可动），因此列表里正好 6 个。

---

## 4. `launch/` —— ROS 启动文件（ROS 1 `roslaunch` 格式）

### `launch/display.launch` —— 只看模型（不需要 Gazebo）
```xml
<param name="robot_description" textfile="$(find <包名>)/urdf/<包名>.urdf" />
<joint_state_publisher_gui> + <robot_state_publisher> + <rviz args="-d .../urdf.rviz">
```
- `robot_description`：把 URDF 全文塞进参数服务器，供 `robot_state_publisher` 计算 TF。
- `joint_state_publisher_gui`：弹出滑条窗口，手动拖动 6 个关节，RViz 里的模型实时动起来
  ——**纯可视化检查 URDF 是否正确**。
- `rviz`：启动 RViz 并加载配置 `urdf.rviz`。
- ⚠️ 注意：这个 `urdf.rviz` 文件**并没有随包导出**（导出的 `-d` 路径指向包根下的 `urdf.rviz`，
  但本目录里没有该文件），所以直接 roslaunch 会因 RViz 配置缺失而报错，
  需要自己保存一份 RViz 配置或去掉 `-d` 参数。
- ⚠️ 另有两个 ROS 1 老问题：包名含**空格**（`7260501-000000-001 TB6-R5-RevA1-URDF`），
  `$(find ...)` 的空格会被拆参；且 `<arg name="model" />` 未使用。
  所以本仓库并不跑 ROS，而是用 `model_import.py` 直接解析这份 URDF（见第 8 节）。

### `launch/gazebo.launch` —— 在 Gazebo 里仿真
- `<include file="$(find gazebo_ros)/launch/empty_world.launch"/>`：起一个空世界。
- `tf_footprint_base`：静态坐标变换 `base_link → base_footprint`（机器人"落地点"惯例）。
- `spawn_model`：用 `-file <URDF> -urdf -model <包名>` 把模型生成到 Gazebo 里。
- `fake_joint_calibration`：向 `/calibrated` 发布 `true`，让控制器认为"已标定"，
  用于**临时绕开标定流程**做测试。
- 因为 URDF 的 `<limit effort="0" velocity="0">` 未填（见第 7 节），
  在 Gazebo 里该模型不会有有效力矩上限，实际只适合做**外形/运动学**演示。

---

## 5. `meshes/` —— 3D 网格（7 个二进制 STL）

由 SolidWorks 逐个零件导出（导出器会**删掉 STL 里的 SolidWorks 头**），
单位与 URDF 一致（米），每个 STL 的坐标系就是对应连杆的坐标系（见下表 `Coordinate System`）。

| 文件 | 大小 | 三角面数 | 对应连杆 | 导出用坐标系 | 含义 |
|---|---|---|---|---|---|
| `base_link.STL` | 986.0 KB | 20192 | `base_link` | `frame1`（Origin_global） | 基座/底座，固定不动，仿真里被固定到世界 |
| `Link1.STL` | 797.3 KB | 16328 | `Link1` | `frame2` | 第 1 轴（腰部回转）连杆 |
| `Link2.STL` | 1393.1 KB | 28530 | `Link2` | `frame3` | 第 2 轴（大臂）连杆，**最重（46.802 kg）** |
| `Link3.STL` | 1320.0 KB | 27032 | `Link3` | `frame4` | 第 3 轴（小臂）连杆 |
| `Link4.STL` | 940.1 KB | 19252 | `Link4` | `frame5` | 第 4 轴（腕部回转）连杆 |
| `Link5.STL` | 1225.2 KB | 25090 | `Link5` | `frame6` | 第 5 轴（腕部摆动）连杆 |
| `ee_Link.STL` | 334.5 KB | 6848 | `ee_Link` | `frame7` | 末端法兰（工具接口面），仿真里 `ee_site` / `tool_site` 就挂在它上面 |

- 在 URDF 里这些网格被**同时**用于 `<visual>`（外观）与 `<collision>`（碰撞体），
  即用高精度网格直接做碰撞检测（仿真中网格面数偏大，`model_import.py` 会另行处理接触问题）。
- 引用形式是 `package://7260501-000000-001 TB6-R5-RevA1-URDF/meshes/<名称>.STL`，
  而本机并没有这层"包名目录"，所以 `model_import.py` 会自动回退解析为 `<本目录>/meshes/...`，
  并在生成 MJCF 时把网格拷到 `assets/meshes/` 改成相对路径（可整目录搬移）。


---

## 6. `urdf/` —— 机器人描述文件

### `urdf/7260501-000000-001 TB6-R5-RevA1-URDF.urdf`（395 行，8.8 KB）
XML 格式，`<robot name="7260501-000000-001 TB6-R5-RevA1-URDF">`，含 **7 个 `<link>` + 6 个 `<joint>`**：

**连杆（link）**：`base_link` → `Link1` → `Link2` → `Link3` → `Link4` → `Link5` → `ee_Link`，串成一条链。
每个 link 含三块信息：
- `<inertial>`：质心 `origin xyz`、`<mass>`、转动惯量 `ixx…izz`（**由 SolidWorks 按材料密度算出的质量属性**）；
- `<visual>` + `<collision>`：同一个 STL 网格 + RGBA 颜色（**碰撞体和外观用同一网格**）；
- `<material>` 颜色来自装配体外观（青绿 / 藕荷 / 土黄 / 淡红等）。

**各连杆质量（URDF 原值，单位 kg）**：

| 连杆 | 质量 | 说明 |
|---|---|---|
| `base_link` | 5.0487 | 基座 |
| `Link1` | 13.203 | 第 1 轴 |
| `Link2` | 46.802 | 第 2 轴（最重） |
| `Link3` | 20.24 | 第 3 轴 |
| `Link4` | 4.6142 | 第 4 轴 |
| `Link5` | 4.7679 | 第 5 轴 |
| `ee_Link` | 1.1773 | 末端法兰 |
| **合计** | **95.8531** | 与 `check_model.py` 自检打印的总质量 **95.85 kg** 一致 |

**关节（joint）**：全部为 `type="revolute"`（绕单轴转动、有上下限的转动副）。

| 关节 | 父 → 子 | origin `xyz`(m) | origin `rpy`(rad) | `<axis>` | `<limit>` lower~upper (rad) |
|---|---|---|---|---|---|
| `joint1` | `base_link` → `Link1` | `0 0 0.0615` | `0 0 0` | `0 0 1` | −3.1415 ~ 3.1415 |
| `joint2` | `Link1` → `Link2` | `0 -0.078 0.0675` | `1.5708 0 0` | `0 0 1` | −3.1415 ~ 3.1415 |
| `joint3` | `Link2` → `Link3` | `-0.442 0 -0.0003` | `0 0 0` | `0 0 1` | **−2.8623 ~ 2.8623** |
| `joint4` | `Link3` → `Link4` | `-0.386 0 -0.0044` | `0 0 0` | `0 0 1` | −3.1415 ~ 3.1415 |
| `joint5` | `Link4` → `Link5` | `0 -0.0543 0.0507` | `1.5708 0 0` | `0 0 1` | −3.1415 ~ 3.1415 |
| `joint6` | `Link5` → `ee_Link` | `0 0.0543 0.0507` | `-1.5708 0 0` | `0 0 **-1**` | −3.1415 ~ 3.1415 |

要点：
- **`joint3` 的限位是 ±2.8623 rad（≈ ±164°）**，是六个关节里唯一不满圈的一个；
  其余都是 ±3.1415 rad（≈ ±180°）。这份限位被 `revA1_spec.py` 的 `LIMITS` 原样采用。
- **`effort="0" velocity="0"`**：URDF 里"力矩上限 / 速度上限"两列没填（SolidWorks 导出时的常见遗漏），
  所以 URDF 编译到 MuJoCo 时 `nu=0`（关节不会动）；`model_import.py` 会自动补
  `<position kp kv ctrlrange forcerange>` 驱动器来修掉这个问题。
- URDF 里**没有** `<transmission>`、`<gazebo>` 标签，也没有 `ee_site` / `tool_site` 之类的 site
  —— 这些是 RevA1-sim 在生成 MJCF 阶段补进去的（见 `revA1_spec.py` 的 `SITES` / `TOOLS`）。

### `urdf/7260501-000000-001 TB6-R5-RevA1-URDF.csv`（8 行 = 表头 + 7 连杆，3.5 KB）
- 与 URDF **同一份数据的表格版**，由导出器 `CSVImportExport.cs` 同时写出。
- 一行 = 一个连杆，列覆盖：质心/质量/惯量、Visual（xyz/rpy/网格/RGBA）、
  Collision（xyz/rpy/网格）、`SW Components`（SolidWorks 零件名，如 `76TB6-R5______-A1-LINK2-1`）、
  `Coordinate System`（`frame1`~`frame7`）、`Axis Name`（`line2`~`line7`）、
  以及该连杆的**关节**信息：`Joint Name` / `Joint Type` / `Joint Origin xyz+rpy` / `Parent` /
  `Joint Axis` / `Limit Effort·Velocity·Lower·Upper` / 标定与动力学等列（后几列为空）。
- **用途**：改机型时在 Excel 里批量核对/回填这些参数，再由 SolidWorks 导出器**重新导入 CSV**
  （`CSVImportExport` 支持读回），避免手工改 URDF。


---

## 7. 这个包在本仓库（RevA1-sim）里怎么被用

| 用到本目录的代码 | 用到的内容 | 说明 |
|---|---|---|
| `revA1_spec.py` | `PACKAGE_ROOT` / `URDF` | 机型参数表。优先用**本目录**（`HERE/"TB6-R5-RevA1"`，目录可整体搬走），找不到才回退仓库根 `RobotSDK/TB6-R5-RevA1`；`LIMITS`、`TOOL_TIP_LOCAL_Z=0.0427`（`ee_Link` 自身沿 +z 长 42.7 mm）等也来自本包的 URDF/网格 |
| `convert_urdf_to_mjcf.py` | `urdf/*.urdf` + `meshes/*.STL` | 命令行入口：URDF → `assets/revA1_arm.xml` + `assets/revA1_scene.xml` + `assets/meshes/` |
| `model_import.py` | 同上 | 转换库：修 `package://` 路径、拷网格、补 actuator、把 root link 包回固定 body、补 site/接触排除 |
| `check_model.py` | 生成的 MJCF | 自检（结构/总质量 95.85 kg/奇异位形/稳定性/工作空间/IK），结果写 `check_model.log` |
| `viewer.py` / `demo_trajectory.py` / `revA1_gui.py` | `assets/revA1_scene.xml` | 可视化、轨迹演示、PySide6 交互控制台 |
| `.vscode/settings.json`（仓库根） | 本目录 | CMake 源目录指向本目录（`cmake.sourceDirectory`），只是给 VS Code 的提示 |

**改机型的标准流程**：
1. 在 SolidWorks 里改装配体 → 用 `sw_urdf_exporter` 重新导出（或直接改本目录的 URDF/CSV）；
2. 覆盖本目录的 `urdf/` 与 `meshes/`；
3. 按需同步 `revA1_spec.py`（限位、工具长度、kp/kv、home 姿态）；
4. 重跑 `python convert_urdf_to_mjcf.py`，再跑 `python check_model.py` 自检。

---

## 8. 注意事项 / 已知问题

1. **包名带空格**：`7260501-000000-001 TB6-R5-RevA1-URDF` 在 ROS 1 的
   `$(find <包名>)`、`package://<包名>/...` 里会被当成多个参数，直接 `catkin_make` / `roslaunch`
   基本跑不通。本仓库因此**不依赖 ROS**，而是由 `model_import.py` 用字符串处理方式解析引用。
2. **`limit effort="0" velocity="0"`**：URDF 未填力矩/速度上限，MuJoCo 直接编译会 `nu=0`
   （关节不动）。转换脚本会补驱动器；若要跑 Gazebo，需要自己补 `<transmission>` 与 `<gazebo>` 标签。
3. **`urdf.rviz` 缺失**：`display.launch` 引用了它，但导出包里没有这个文件。
4. **碰撞体 = 外观网格**：面数较高（单件 0.7~2.9 万三角面），且 `base_link`/`Link1` 贴合面会
   产生持续接触；`model_import.py` 已做处理（把 root geom 包回固定 body 并做接触排除）。
5. **`RDM.xml` 不是 ROS 资源**：它只作为驱动器（Techservo RDM，CiA402／EtherCAT）的
   参数与对象字典参考资料，代码中没有引用；删掉它不影响仿真。
6. **`export.log` 里的历史记录**：日志包含多次试导出的路径（`...REV`、`...RevA1_urdf` 等），
   只有 `...RevA1-URDF` 那一份与当前目录内容对应；排查问题时以最后 100 行左右为准。
7. **本包只含机械/电气描述**：真机通信（UDP 6001 广播状态、UDP 6501 任务信号、HTTP 状态）
   在 `robot_link.py` 里，不在本目录内。

---

## 9. 一句话速查

| 想找什么 | 看哪个文件 |
|---|---|
| 机器人长什么样、质量/惯量多少 | `urdf/*.urdf`（数值）、`meshes/*.STL`（外形） |
| 关节限位、关节轴方向、父子关系 | `urdf/*.urdf`（`<joint>` 段）或 `urdf/*.csv`（表格） |
| 好用 Excel 批量核对参数 | `urdf/*.csv` |
| 在 RViz / Gazebo 里打开 | `launch/display.launch`、`launch/gazebo.launch` |
| 关节名清单 | `config/joint_names_*.yaml` |
| 驱动器（RDM）对象字典 / PDO 映射 | `RDM.xml` |
| 这份包是怎么被导出的 | `export.log` |

