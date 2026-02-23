# README for Waveform-aware Routing Protocol for Airborne Networks (MCA_Routing only code)

This code has been adapted from [SPEproject-802.11DCFsim](https://github.com/riccardomicheletto/SPEproject-802.11DCFsim). It provides a discrete event simulator for the **802.11 CSMA/CA DCF based routing protocol with adaptive MCS-SNR capabilities**. The simulator is written in Python and it leverages on the [Simpy](https://simpy.readthedocs.io/en/latest/) framework. The code is developed on Ubuntu 20.04 LTS using Python version 3.13. To install the dependencies required to run the code, execute the following command: ``pip install -r requirements.txt``. This will install packages such as Simpy, Scipy, Pandas, Matplotlib and Networkx.

### Usage

After installing Simpy, run the main file to execute the simulator with `python main.py [seed]`. You will be prompted to enter some config options. Default values will be prefilled which you can modify. For advanced users: Modify parameter(s) within config files within ./config/cfg to modify the simulation settings. The default config file is ./config/cfg/default.yaml which can be used as a template for creating config files. A config file can be passed in using the --config option. The interactive prompts can be suppressed using --no-input, in case you only want to use config files. Run `set_schema.py` to set up config variables, and JSONSchema vscode settings. After this, hovering keys in the configuration file should display an explanation of the configuration keys. Otherwise, this information can be viewed in `schema.yaml`. A more detailed README specifically for the config is available in `./config/config.md`

To run multiple simulations use `RunAll.py`. Use -h to get help. Pass in the number of runs and the path to the config you want to use. It will use the seeds from `seed_numbers.txt` unless you pass in another file using --seeds. Use --parallel to control the number of runs that can be running in parallel at once.

To get insights of the code during run-time, set `PRINT_LOGS` to **True.**

**ROUTE_DETAILS** variable in parameters.py stores the flow source-destination pairs, priority, initial routes and their MCS index, and a flag to indicate if current route meets quality thresholds (useful insight for MCA-aware routing). For each flow, the flow source node uses the MCA features proposed for MCA-AODV and MCA-OLSR protocols (see folder with **MCA_Routing_** prefix). Use `ENABLE_Q_MNGMNT` in the config to enable/disable queue management. When `USE_FLOW_PRIORITY` in the config is disabled, all flows have the same priority of 1. Else, flow priority is equal to the route index. **MCA_Routing_** provides a trade-off for route length (in hops), number of interfering links and route capacity vs. route cost. It also filters out poor-quality routes that are either short-lived or have a high estimated-time-to-destination or both.

The flow source generates the packets at a constant bitrate depending on the config variable `PAYLOAD_DATA_RATE`. We have added the support for packet queuing at each node, which would allow packet inspection (e.g., to see packet service time) and manipulation (e.g., packet drop due to expiry and/or buffer overflow). Intermediate node forwards the received packet until the packet reaches the flow destination node or is dropped due to expiry, route unavailability, exceeding retry limit or buffer overflow.

Finally, the stat logs are stored in the folder specified by `RESULTS_FOLDER` in the config. It contains **NetworkGraphs** folder which stores the network topology observed at different intervals, and a gif file showing the changes in the network topology with time. If **./GenerateResult.sh** script in the same folder as main.py is used, the results are moved to **./Results/** folder after the simulation completes.
**ComplieResults.py** automates result extraction. Run it after the simulation completes. It is able to handle a folder with result folders (1 level nested). To use it for a group run a command like the following: `for d in PATH_TO_MY_RESULTS/*/; do python ComplieResults.py "$d"; done`

### How to Set Up Different Topologies

Different topologies can be set up by modifying what `STATIC_TOPOLOGY_PATH` in the config file points to, if you are using a **static topology**. The file should contain a list of positions representing latitude and longitude like: "1000, 1000" on each line. The nodes are set up starting with ID 0. If you wish to start with ID 1, you can setup a dummy node by placing "-9999, -9999" in the first line. Relative values are also accepted. A line containing "+1000, -500" will be parsed as adding or subtracting from the previous line's values if available. To input absolute positions which are negative use "(-1000), 500". However, it is recommended to translate your positions to use positive values.

For the above static topology, the route should be set up using `FORCED_ROUTES`, which is a list which contains list of source destination pairs, e.g. [[1, 5], [7, 9]]. The MCS for data should be changed using `DEFAULT_DATA_MCS_INDEX`. To change a specific node's MCS, change `FORCED_ROUTE_MCS` to a nested dictionary containing the route ID (starts at 0) as the key, and as the value, a dictionary containing the Node ID as the key and MCS as the value.

For a mobile scenario, a topology will be simulated using the mobility model. Update `NUMBER_OF_BS`, `NUMBER_OF_NODES`, and `AREA_X`. Also, to place targets add the locations to `TARGET_LOCATION` as a list of positions (as a list of length 2), i.e. [[5000, 5000], [2500, 7500]]. Modify `BS_SCHEMES` and `CHOSEN_BS_SCHEME` to select the locations of the base stations. The scheme corresponding to the chosen BS scheme will be used placing base stations in the order of the scheme until the number specified is reached. Routing will automatically create routes from the first node to reach a target to the best base station, so forced routes are not needed.

### Simulation Speedup

There are two additional actions you can take to speed up the performance of the simulation.

1. Run `python3 setup.py build_ext --inplace` to compile the Cython code in `phero_c.pyx`, which will create C versions of functions that will be imported and used in place of Python functions automatically if available.
2. Run `compile_packages.py` to create compiled versions of packages installed which are purely written in Python. Follow the directions at the top of the file to make sure you are using the compiled versions.

### Code Layout

The code is structured so that there are 5 main components in the running of the simulation: Mobility Model (BSCAP), Ether, Node, MAC, Phy. There are also 3 helper components: RouteFunctions, Stats, load_config.

The Mobility Model (BSCAP) computes node trajectories which will be used if we are running a mobile scenario, and it runs after nodes are initialized, and continues to run after data transmission begins.

Ether is the shared wireless medium for all nodes which sends PhyPackets (with begin/end flags) to listening nodes. There is one ether object which is created at the start of the simulation which is referenced by all of the nodes.

Node generates packets if it is a source node. Each node in the route enqueues generated and received packets for transmission, handles packet aggregation, and forwards packets to the downstream node on the route if it is an intermediate node in a route.

MAC manages channel access using backoff and NAV, and handles RTS/CTS and ACK logic, and retransmissions, as well as power control.

Phy performs physical sensing and transmission using Ether, encapsulates MacPackets into PhyPackets and computes SINR on reception, and delivers received packets upward to MAC based on decoding success.

*The above three classes (Node, MAC, Phy) can access each other. Node contains a reference to MAC which in turn contains a reference back to Node and a reference to Phy, while Phy has a reference back to MAC.*

RouteFunctions contains the code to handle route selection, and syncs trajectories calculated by the mobility model to all nodes to be used in routing. It also contains functions which create images of the topology and store them.

Stats records node and flow level statistics which are used for post-simulation level analysis.

### Code Walkthrough

The following is a timeline of a typical packet.

1. In main.py, config variables are loaded, ether is created, all nodes are set up.
2. If MOBILE_SCENARIO is True, it will run the BSCAP mobility model to calculate the trajectories of the nodes.
3. At PKT_GENERATION_START_SEC, data transmission begins.
4. In each flow, the source Node of the flow begins generating packets and enqueuing them.
5. EnqueuePacket() checks expiry, then places the packet in the queue and makes sure keepSending() is running if it should be.
6. keepSending() sends packet while the queue is not empty.
   If MAC already has a packet, it waits, then if USE_PACKET_AGGREGATION is True, it aggregates packets (if it can), then sends a packet to MAC.
7. Mac's send() waits for itself to finish sensing for another packet and Phy to finish sending.
   Then, it starts a waitIdleAndSend() process for the MacPacket, which is a wrapper around the packet ID string, it wants to send.
8. waitIdleAndSend() waits while its backoff is > 0 or it has a NAV set.
   Then it calculates if its MacPacket will result in a CTS or ACK timeout.
   It also calculates power to send the packet at and the NAV duration, which will be different between the CTS and ACK paths.
   It sets the NAV at its neighbors and sends the MacPacket to Phy, and begins the waitAck() process for the MacPacket.
9. Phy interrupts its listen() process with the MacPacket it got from Mac and calls encapsulateAndTransmit().
10. encapsulateAndTransmit() wraps the MacPacket with a PhyPacket, and calls ether.transmit() to send the packet to nodes that are listening.
    It sends the same PhyPacket along with beginOfPacket and endOfPacket flags a number of times depending on the size of the packet.
11. The destination node's Phy detects those transmissions in its listen() function. It does physical sensing which freezes backoff,
    if the channel is busy. When it detects an endOfPacket for a packet it was receiving, it calculates the SINR with computeSinr().
    computeSinr() maps each packet of a combined packet to a bucket of instantaneousSINR values and takes the minimum for each.
    listen() then creates a process for handleReceivedPacket() at the destination's Mac.
12. handleReceivedPacket() receives the packet and enqueues it at Node for forwarding if necessary. It then creates an ACK MacPacket,
    with flags for whether each packet was received, and sends it to its Phy. That ACK goes through the same sending process in Phy.
    When it arrives at the source's handleReceivedPacket(), the waitAck() process for that packet is interrupted.
13. waitAck() if it was interrupted after it timed out, considers all its packets unsuccessful.
    Otherwise, it divides successful and unsuccessful packets, and logs successes and retransmits unsuccessful packets.
    It increases their retransmission counter, waits for Phy sending and its sensing, and calls waitIdleAndSend() for the new MacPacket.

### Control Packets

In our implementation, RTS/CTS frames are not physically transmitted over the medium; however, their functionalities are emulated because a transmitter node knows its own as well as its receiver's one hop neighbors.

When a transmitter node intends to send an RTS, it first checks whether the intended receiver node has its NAV set (i.e., it is busy). A set NAV would prevent the receiver from replying with a CTS. If the receiver is busy, the transmitter sends a dummy packet (does not include data) whose payload length matches that of an RTS. This allows the transmitter's 1‑hop neighbors to receive the packet and update their interference state accordingly. Note that these neighbors discard the packet at the physical layer after using it to update interference and do not pass it to the MAC layer. In this scenario, the transmitter node directly sets the NAV at its 1‑hop neighbors. No NAV is set at the receiver’s 1‑hop neighbors because the receiver does not send (or emulate sending) any CTS packet. The CTS timeout value is used as the NAV duration at transmitter's one hop neighbors.

If the receiver is not busy (i.e., no active NAV), the transmitter piggybacks the effective RTS and CTS durations onto the actual payload. It also includes the appropriate SIFS intervals to match the timing that would occur in a real RTS/CTS exchange before data transmission. The NAV is computed based on the ACK timeout, and the transmitter updates the NAV at both its own 1‑hop neighbors and the receiver’s 1‑hop neighbors.

In our simulation, the transmission delay is adjusted based on the MCS used for control (RTS/CTS/ACK) and data packets as well as based on payload (including packet aggregation, if any). 

### Power Control

In our power control scheme, each transmitter node can adapt its power which changes its transmission range. To account for this in our implementation, the nodes which can detect and/or decode the packet are computed based on the transmission power, the MCS used, and their distance from the transmitter node.

The power used by the nodes is calculated as the following.

RTS Power:

* P_rts_default is MCS 0 (`CONTROL_MCS_INDEX`) at D_max + interference at the receiver (which we are able to get the same way we get gps coordinates).
* P_rts_datamcs is data's MCS at actual link length (distance between transmitter receiver pair) + interference at receiver.
* Whichever of P_rts_default or P_rts_datamcs is higher.

CTS Power:

* P_cts is MCS 0 (`CONTROL_MCS_INDEX`) at D_max + interference at the transmitter.

Data Packet Power:

* P_data is data's MCS at actual link length + interference at receiver.

ACK Power

* P_ack is MCS 0 (`CONTROL_MCS_INDEX`) at D_max + interference at the transmitter.

  RTS, CTS, Data, and ACK powers capped at maximum transmission power.

### Channel Sensing

Physical sensing is done in Phy, and is used to determine whether the channel is busy. It stores the power that a node receives from its neighbors in each timeslot as interference. When the power required to overcome that interference at the node rises above a threshold (here threshold is determined by the power used to transmit to the maximum distance (D_max) at the given MCS with no interference, multiplied by `MAX_POWER_MULTIPLE`), it determines that the channel is busy. When Phy determines that the channel is busy, MAC will freeze its backoff (the timeout counter will not decrement even while MAC is waiting).

### Packet Aggregation

When `USE_PACKET_AGGREGATION` is True and the MCS being used is above 0, Node will attempt to aggregate packets if it is able to. It will do this by concatenating the individual packet IDs with `_COMBINED_`. If `ENABLE_INDIVIDUAL_PKT_SINR` is True, the ACK packet returned by the receiver will contain flags for each individual packet communicating whether it was accepted by the receiver or not. If the ACK is not received by the sender before NAV timeout, it will assume all packets were not accepted. If only some of the packets were not accepted, the packets that were not accepted will be retransmitted. In this retransmission, currently no new packets are added to the aggregated packet, only the unaccepted packets are used to create a new MacPacket, and retransmitted.

### How BSCAP Base Stations Affect Routing

When `MOBILE_SCENARIO` is being used, and there are no ` FORCED_ROUTES`, routes will be created from the first node which finds a target to a base station (see function in RouteFunctions.py). The base station is selected based on the `ROUTING_COST_TYPE`. If it is 2, the base station which is the least hops from the source will be chosen as the destination. If it is 3, the base station which is the shortest euclidean distance will be chosen as the destination. For both 2 and 3, the route with least hops to the destination base station will be chosen. If it is 1, the following routing cost formula will be used (see the config for the weights):

`(parameters.W1 * (path['HC'] / min_HC) + parameters.W2 * (path['IL'] / (min_IL + parameters.ALPHA * path['IL']))) * max_data_rate / path['Max Data Rate Supported']`

In this case, a virtual gateway node is created to act as the destination, which connects to all base stations but no other nodes; cost of the link from gateway node to each base station is 0. A source node finds route to the gateway node through each base station where a base station acts as a penultimate node on the route (i.e., it is the last link from the base station to the gateway node with 0 cost). The route cost is calculated from the source node to the gateway node by considering each base station as the penultimate node; the route with the least cost is selected. The gateway node is then removed from the end of the route, resulting in a route from the source node to the least cost base station.

### Important Changes and Observations

1. Decrementing timeout value by 1 in KeepSending() in mac.py file incurs a huge computational delay. Therefore, we have used a larger value (STEP_SIZE) to speed up the simulation.
2. Using hop count as the route selection metric results in a shortest hop count route, consisting of edge nodes. Therefore, the capacity of the route is lower since edge nodes may not achieve higher SINR value required to support higher MCS index.
3. ~~Links with a higher capacity are selected when the link cost is used as the routing metric. This is possible at lower node distances, when the transmission power is kept constant. Therefore, this results in a higher-capacity but longer-hop route, which increases congestion along the route, and therefore, may not be suitable either.~~
4. ~~Queue management feature is a MCA feature. **HC_Only** scheme does not support it.~~
5. The use of queue management involves preemptively discarding packets that might not reach the destination node based on their current packet survivability score. If not configured properly, this may reduce throughput due to unneccessarily dropping the packets that could have reached the destination node before their expiry.
6. ~~**ENABLE_PACKET_REINSERTION** enabled will reinsert only those packets at the source node which were dropped at the intermediate nodes due to sudden route change. It does not consider other packet drop issues such as exceeding retry limit or buffer overflow at intermediate nodes (not at the source node since its buffer is already full).~~
