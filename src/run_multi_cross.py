from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import random, os

def subprocess(command: str, index: int, max: int):
	print(f'Starting execution {index}/{max}')
	result = os.system(command)
	return f'Finished: {index}/{max} with result: {result}'


if __name__ == '__main__':
	topology_files = [
		"./nodeTrajectories/positions/crossFlow_7_Small.txt", 
		"./nodeTrajectories/positions/crossFlow_7_Medium.txt", 
		"./nodeTrajectories/positions/crossFlow_7_Large.txt"]
	results_folder = './../Results_Cross/'
	
	# Set target locations as the first N nodes in the topology (because the last is the base station)
	num_targets = 2
	targets = [[[6000, 4500],[4500, 6000]], [[6000, 3000],[3000, 6000]], [[6000, 0],[0, 6000]]]

		
	node_distances = [500, 1000, 2000] #Small medium large
	flow_data_rates = [[2.0,2.0], [4.0,4.0], [8.0,8.0]]
	forced_routes = [[0,12], [6,11]]
	forced_mcs = [
		{
			0: {0:0, 1:0, 2:0, 3:1, 4:0, 5:0, 12:0}, 
   			1: {6:0, 7:0, 8:0, 3:1, 9:0, 10:0, 11:0}
		},
		{
			0: {0:0, 1:0, 2:1, 3:1, 4:1, 5:0, 12:0}, 
   			1: {6:0, 7:0, 8:1, 3:1, 9:1, 10:0, 11:0}
		},
		{
			0: {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 12:0}, 
   			1: {6:0, 7:1, 8:1, 3:1, 9:1, 10:1, 11:0}
		},
		{
			0: {0:1, 1:1, 2:1, 3:0, 4:1, 5:1, 12:1}, 
   			1: {6:1, 7:1, 8:1, 3:0, 9:1, 10:1, 11:1}
		},
		{
			0: {0:1, 1:1, 2:0, 3:0, 4:0, 5:1, 12:1}, 
   			1: {6:1, 7:1, 8:0, 3:0, 9:0, 10:1, 11:1}
		},
		{
			0: {0:1, 1:0, 2:0, 3:0, 4:0, 5:0, 12:1}, 
   			1: {6:1, 7:0, 8:0, 3:0, 9:0, 10:0, 11:1}
		},
		{
			0: {0:0, 1:0, 2:0, 3:1, 4:1, 5:0, 12:0}, 
   			1: {6:0, 7:0, 8:0, 3:1, 9:1, 10:0, 11:0}
		},
		{
			0: {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 12:0}, 
   			1: {6:0, 7:0, 8:0, 3:1, 9:1, 10:1, 11:0}
		}
	]
	mcs_scheme_names = ['hotspot_1', 'hotspot+1_1', 'hotspot+2_1', 'hotspot_0', 'hotspot+1_0', 'hotspot+2_0', 'hotspot+down_1', 'hotspot+down2_1']
	num_base_stations = 2
		
	# Load types

	total_count = len(topology_files) * len(flow_data_rates) * len(forced_mcs)
	count = 1
	overrides = {}
	execution_commands = []
	for topology, target, node_distance in zip(topology_files, targets, node_distances):
		for data_rate in flow_data_rates:
			for mcs_name, mcs_dict in zip(mcs_scheme_names, forced_mcs):
				output_folder = f'{results_folder}{topology.split("/")[3].split(".")[0]}_{node_distance}_{str(data_rate[0]).replace(".","-")}_{mcs_name}/'
				# print(f"Simulating {count}/{total_count}: {output_folder}")

				# do overrides
				overrides['RESULTS_FOLDER'] = f'{output_folder}'
				overrides['STATIC_TOPOLOGY_PATH'] = f'{topology}'
				overrides['MOBILITY.TARGET_LOCATION'] = f'{target}'
				overrides['MAXIMUM_ROUTING_RANGE'] = str(node_distance+100)
				overrides['PAYLOAD_DATA_RATE'] = f'{data_rate}'
				if num_base_stations != 1:
					overrides['NUMBER_OF_BS'] =  f'{num_base_stations}'
					if node_distance == 500:
						overrides['MOBILITY.BS_SCHEME'] = [[6000, 7500],[7500, 6000]]
					elif node_distance == 1000:
						overrides['MOBILITY.BS_SCHEME'] = [[6000, 9000],[9000, 6000]]
					elif node_distance == 2000:
						overrides['MOBILITY.BS_SCHEME'] = [[6000, 12000],[12000, 6000]]
				if len(forced_routes) > 0:
					overrides['FORCED_ROUTES'] =  f'{forced_routes}'
				overrides['FORCED_ROUTE_MCS'] = mcs_dict

				default_args = f'--config .\\config\\cfg\\daisy_chain.yaml --schema .\\config\\schemas\\daisy_schema.json --no-input'
				override_string = ''
				for key,value in overrides.items():
					override_string += f'--{key}="{value}" '
				execution_commands.append((
					f'uv run ./main.py {random.randint(1,1000)} {default_args} {override_string}'
					f' && uv run ./CompileResults.py {overrides["RESULTS_FOLDER"]}', count))
				count += 1
				
	max_count = len(execution_commands)
	with ProcessPoolExecutor() as executor:
		futures = [
			executor.submit(subprocess, comm, index, max_count)
			for comm,index in execution_commands
		]
		for f in as_completed(futures):
			print(f.result())


