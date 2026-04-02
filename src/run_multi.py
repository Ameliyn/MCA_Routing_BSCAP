from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import random, os

def subprocess(seed: int, default_args: str, override_string: str, results_folder):
	print(f'uv run ./main.py {seed} {default_args} {override_string}')
	os.system(f'uv run ./main.py {seed} {default_args} {override_string}')
	os.system(f'uv run ./CompileResults.py {results_folder}')

if __name__ == '__main__':
	topology_files = ["./nodeTrajectories/positions/daisy_chain_small.txt", "./nodeTrajectories/positions/daisy_chain_medium.txt", "./nodeTrajectories/positions/daisy_chain_large.txt"]
	
	# Set target locations as the first node in the topology (because the last is the base station)
	targets = []
	for topo in topology_files:
		with open(f'{topo}', 'r') as f:
			targets.append(f'[{f.readline().strip()}]')
		
	max_routing_ranges = [600, 1200, 2400] #Small medium large
	flow_data_rates = [1.5, 2.0, 4.0, 8.0]
	MCS_types = [1,0]
		
	# Load types

	total_count = len(topology_files) * len(flow_data_rates) * len(MCS_types)
	count = 1
	overrides = {}
	for topology, target, routing_range in zip(topology_files, targets, max_routing_ranges):
		for data_rate in flow_data_rates:
			seeds = [random.randint(1,1000), random.randint(1,1000)]
			override_strings = []
			results_folders = []
			for mcs in MCS_types:
				output_folder = f'./../Results_Multi_N100/{topology.split("/")[3].split(".")[0]}_{routing_range}_{str(data_rate).replace(".","-")}_{mcs}/'
				print(f"Simulating {count}/{total_count}: {output_folder}")
				count += 1

				# do overrides
				overrides['RESULTS_FOLDER'] = f'{output_folder}'
				overrides['STATIC_TOPOLOGY_PATH'] = f'{topology}'
				overrides['MOBILITY.TARGET_LOCATION'] = f'[{target}]'
				overrides['MAXIMUM_ROUTING_RANGE'] = str(routing_range)
				overrides['PAYLOAD_DATA_RATE'] = f'[{data_rate}]'
				overrides['DEFAULT_DATA_MCS_INDEX'] =  f'{mcs}'

				default_args = f'--config .\\config\\cfg\\daisy_chain.yaml --schema .\\config\\schemas\\daisy_schema.json --no-input'
				override_string = ''
				for key,value in overrides.items():
					override_string += f'--{key}="{value}" '
				override_strings.append(override_string)
				results_folders.append(overrides['RESULTS_FOLDER'])

			with ProcessPoolExecutor(max_workers=2) as executor:
				futures = [
					executor.submit(subprocess, seed, default_args, override, res_fold)
					for seed, override, res_fold in zip(seeds, override_strings, results_folders)
				]
				for f in as_completed(futures):
					print(f.result())


