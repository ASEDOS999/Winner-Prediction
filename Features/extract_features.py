import bz2
import json
import pandas
import collections
import argparse


def last_value(series, times, time_point=60*5):
	values = [v for t, v in zip(times, series) if t <= time_point]
	return values[-1] if len(values) > 0 else 0

def filter_events(events, time_point=60*5):
	return [event for event in events if event['time'] <= time_point]

def extract_match_features(match, time_point=None):
	extract_items_time = [
		(41, 'bottle'),
		(45, 'courier'),
		(84, 'flying_courier'),
	]
	
	extract_items_id_count = pandas.read_csv('../data/dictionaries/items.csv')['id'].tolist()

	feats = [
		('match_id', match['match_id']),
		('start_time', match['start_time']),
		('lobby_type', match['lobby_type']),
	]
	
	# player features
	
	times = match['times']
	for player_index, player in enumerate(match['players']):
		player_id = ('r%d' % (player_index+1)) if player_index < 5 else ('d%d' % (player_index-4))
		
		feats += [
			(player_id + '_hero', player['hero_id']),
			(player_id + '_level', max([0] + [entry['level'] for entry in filter_events(player['ability_upgrades'], time_point)])),
			(player_id + '_xp', last_value(player['xp_t'], times, time_point)),
			(player_id + '_gold', last_value(player['gold_t'], times, time_point)),
			(player_id + '_lh', last_value(player['lh_t'], times, time_point)),
			(player_id + '_kills', len(filter_events(player['kills_log'], time_point))),
			(player_id + '_deaths', len([
				1
				for other_player in match['players']
				for event in filter_events(other_player['kills_log'], time_point)
				if event['player'] == player_index   
			])),
			(player_id + '_items', len(filter_events(player['purchase_log'], time_point))),
		]
	
	# first blood
	first_blood_objectives = [obj for obj in match['objectives'] if obj['type'] == 'firstblood']
	fb = first_blood_objectives[0] if len(first_blood_objectives) > 0 else {}
	feats += [
		('first_blood_time', fb.get('time')),
		('first_blood_team', int(fb['player1'] >= 5) if fb.get('player1') is not None else None),
		('first_blood_player1', fb.get('player1')),
		('first_blood_player2', fb.get('player2')),
	]
	
	# team features
	radiant_players = match['players'][:5]
	dire_players = match['players'][5:]
	
	for team, team_players in (('radiant', radiant_players), ('dire', dire_players)):
		for item_id, item_name in extract_items_time:
			item_times = [
				entry['time']
				for player in team_players
				for entry in filter_events(player['purchase_log'], time_point)
				if entry['item_id'] == item_id
			]
			first_item_time = min(item_times) if len(item_times) > 0 else None
			feats += [
				('%s_%s_time' % (team, item_name), first_item_time)
			]
			
		for item_id in extract_items_id_count:
			item_count = sum([
				1
				for player in team_players
				for entry in filter_events(player['purchase_log'], time_point)
				if entry['item_id'] == item_id
			])
			feats += [
				('%s_%d_count' % (team, item_id), item_count)
			]
			
		team_wards = filter_events([
			entry
			for player in team_players
			for entry in (player['obs_log'] + player['sen_log'])
		], time_point)
		
		feats += [
			('%s_first_ward_time' % team, min([entry['time'] for entry in team_wards]) if len(team_wards) > 0 else None),
		]

	if 'finish' in match:
		finish = match['finish']
		feats += [
			('duration', finish['duration']),
			('radiant_win', int(finish['radiant_win'])),
		]

	return collections.OrderedDict(feats)


def iterate_matches(matches_filename):
	with bz2.BZ2File(matches_filename) as f:
		for n, line in enumerate(f):
			match = json.loads(line.decode())
			yield match
			if (n+1) % 1000 == 0:
				print ('Processed %d matches' % (n+1))


def create_table(matches_filename, time_point, stop):
	df = {}
	fields = None
	N = 0
	for match in iterate_matches(matches_filename):
		N += 1
		features = extract_match_features(match, time_point)
		if fields is None:
			fields = features.keys()
			df = {key: [] for key in fields}    
		for key, value in features.items():
			df[key].append(value)
		if N > stop:
			break
	df = pandas.DataFrame.from_records(df).ix[:, fields].set_index('match_id').sort_index()
	return df


if __name__ == '__main__':
	print('Name of output file:')
	name = input()
	print('Number of processed lines:')
	stop = int(input())
	print('Please, wait...')
	parser = argparse.ArgumentParser(description='Extract features from matches data')
	parser.add_argument('--time', type=int, default=5*60)
	args = parser.parse_args()
	
	features_table = create_table(r'../matches.jsonlines.bz2', args.time, stop)
	features_table.to_csv(name)
